import asyncio
import logging
import os
from urllib.parse import urljoin
import aioboto3
import aiobotocore
import io
import json

from aiogram import Bot, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook


API_TOKEN = os.environ['BOT_TOKEN']

# webhook settings
WEBHOOK_HOST = os.environ['WEBHOOK_HOST_ADDR']
WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

# webserver settings
WEBAPP_HOST = '0.0.0.0'  # or ip
WEBAPP_PORT = os.environ['PORT']

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


# parameters to connect to AWS
REGION_NAME = os.environ['REGION']
AWS_ACCESS_KEY_ID = os.environ['ACCESS_KEY']
AWS_SECRET_ACCESS_KEY = os.environ['SECRET_KEY']
AWS_ACCESS_INFO_DICT = {
    'region_name': REGION_NAME,
    'aws_access_key_id': AWS_ACCESS_KEY_ID,
    'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
}
AWS_ROLE = os.environ['ROLE']
AWS_DEFAULT_BUCKET = os.environ['DEFAULT_BUCKET']

MESSAGE_TO_CONTACT_SUPPORT = os.getenv('MESSAGE_TO_CONTACT_SUPPORT', 'please communicate with the author')

CYCLEGAN_STYLENAME_TO_MODELNAME_DICT = {
    "Monet": "style_monet",
    "Van Gogh":"style_vangogh",
    "Cezanne": "style_cezanne",
    "Ukiyo-e": "style_ukiyoe",
}


def get_stylechoice_markup():
    '''
    Get ReplyKeyboardMarkup with choice of predefined styles corresponding to pretrained CycleGAN models
    '''

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True, one_time_keyboard=True)
    markup.row_width = 2
    style_name_list = list(CYCLEGAN_STYLENAME_TO_MODELNAME_DICT.keys())
    markup.add(*style_name_list)
    return markup


async def _read_s3_obj_as_bytes(s3_obj):
    '''
    Read object from S3 as bytes, i.e. raw contents of some file corresponding to this object
    '''

    s3_obj_response = await s3_obj.get()
    s3_obj_body = s3_obj_response['Body']
    return await s3_obj_body.read()


async def _read_s3_obj_as_string(s3_obj):
   '''
   Read object from S3 as string, i.e. contents of some text file corresponding to this object
   '''

   s3_obj_content = await _read_s3_obj_as_bytes(s3_obj)
   return s3_obj_content.decode("utf-8")


def _get_prefix(id):
   '''
   Get prefix in the bucket defined by AWS_DEFAULT_BUCKET for data corresponding to a certain chat with chat_id given by id
   '''

   return f'userdata/{id}/'


async def _get_userdata_for_model_inputs(id, assume_userdata_exist=False):
    '''
    Get info on model inputs for some chat with chat_id given by id
 
    by default this function checks whether the corresponding folder exists and if not, then creates it
    (not done if assume_userdata_exist equals to True), after this returns contents of files
    content.txt and style.txt (or None if some of them do not exist), these files contain
    URL to uploaded content image and either URL to uploaded style image or style name for predefined styles,
    respectively  
    '''

    prefix = _get_prefix(id)
    content_img_url = None
    style_img_url_or_name = None
    async with aioboto3.resource("s3", **AWS_ACCESS_INFO_DICT) as s3:
        bucket = await s3.Bucket(AWS_DEFAULT_BUCKET)
        if assume_userdata_exist:
            is_userdata = True
        else:
            async for obj in bucket.objects.filter(Prefix=prefix):
                is_userdata = True
                break
            else:
                is_userdata = False
        if not is_userdata:
            await bucket.put_object(Key=prefix)
        else:
            async for obj in bucket.objects.filter(Prefix=f'{prefix}content.txt'):
                content_img_url = await _read_s3_obj_as_string(obj)
                break
            async for obj in bucket.objects.filter(Prefix=f'{prefix}style.txt'):
                style_img_url_or_name = await _read_s3_obj_as_string(obj)
                break
    return content_img_url, style_img_url_or_name


async def _get_userdata_for_model_outputs(id, assume_userdata_exist=False, increment_counter=False):
    '''
    Get info on model outputs for some chat with chat_id given by id

    by default this function checks whether the corresponding folder exists and if not, then creates it
    (not done if assume_userdata_exist equals to True), after this returns URL of current resulting image
    (or None in the case no resulting image is expecting), this S3 path is based on the counter stored
    in result_counter.txt file put on S3
    if increment_counter is False (default), then this function returns S3 path based on the current value
    of the counter, otherwise it increments the counter and then returns the respective S3 path, the latter
    is done to prevent conflicts between resulting images obtained by already cancelled requests
    and resulting images for new requests, that is some resulting image is to send to Telegram iff
    current resulting image S3 path coincides with the one obtained from SageMaker (in the case when
    we cancel some request or start a new one, we update the counter)

    '''

    prefix = _get_prefix(id)
    result_counter = None
    result_img_url = None
    async with aioboto3.resource("s3", **AWS_ACCESS_INFO_DICT) as s3:
        bucket = await s3.Bucket(AWS_DEFAULT_BUCKET)
        if assume_userdata_exist:
            is_userdata = True
        else:
            async for obj in bucket.objects.filter(Prefix=prefix):
                is_userdata = True
                break
            else:
                is_userdata = False
        if not is_userdata:
            await bucket.put_object(Key=prefix)
            if increment_counter:
                await bucket.put_object(Key=f'{prefix}result_counter.txt', Body="1".encode("utf-8"))
        else:
            async for obj in bucket.objects.filter(Prefix=f'{prefix}result_counter.txt'):
                result_counter = await _read_s3_obj_as_string(obj)
                break
            if result_counter is not None:
               result_counter = int(result_counter) if result_counter.isdigit() else None
            if increment_counter:
                if result_counter is None:
                    result_counter = 0
                result_counter += 1
                await bucket.put_object(Key=f'{prefix}result_counter.txt', Body=str(result_counter).encode("utf-8"))
    if result_counter is not None:
        result_img_url = f's3://{AWS_DEFAULT_BUCKET}/{prefix}result{result_counter}.jpg'
    return result_img_url


async def _put_userdata_for_model_inputs(id, input_file_name, input_value, assume_userdata_exist=False):
    '''
    Put some string given by input_value into some input file stored on S3 with info on one of model inputs
    for some chat with chat_id given by id

    by default this function checks whether the corresponding folder exists and if not, then creates it
    (not done if assume_userdata_exist equals to True), after this it writes input_value into the file,
    possibly overwriting the old contents, input_file_name may be equal to content.txt or style.txt
    '''

    prefix = _get_prefix(id)
    async with aioboto3.resource("s3", **AWS_ACCESS_INFO_DICT) as s3:
        bucket = await s3.Bucket(AWS_DEFAULT_BUCKET)
        if assume_userdata_exist:
            is_userdata = True
        else:
            async for obj in bucket.objects.filter(Prefix=prefix):
                is_userdata = True
                break
            else:
                is_userdata = False
        if not is_userdata:
            await bucket.put_object(Key=prefix)
        await bucket.put_object(Key=f'{prefix}{input_file_name}', Body=input_value.encode("utf-8"))


async def _clean_userdata(id, assume_userdata_exist=False):
    '''
    Clean all the files (save result_counter.txt) from user data folder in S3 for some chat with chat_id given by id

    by default this function checks whether the corresponding folder exists and if not, then creates it
    (not done if assume_userdata_exist equals to True), after this it cleans the folder
    '''

    prefix = _get_prefix(id)
    async with aioboto3.resource("s3", **AWS_ACCESS_INFO_DICT) as s3:
        bucket = await s3.Bucket(AWS_DEFAULT_BUCKET)
        if assume_userdata_exist:
            is_userdata = True
        else:
            async for obj in bucket.objects.filter(Prefix=prefix):
                is_userdata = True
                break
            else:
                is_userdata = False
        if not is_userdata:
            await bucket.put_object(Key=prefix)
        async for obj in bucket.objects.filter(Prefix=prefix):
            if obj.key != prefix and obj.key != f'{prefix}result_counter.txt':
                await obj.delete()



@dp.message_handler(commands=['start', 'cancel'])
async def start_or_cancel_handler(message: types.Message):
    '''
    Handle start or cancel commands
    '''

    content_img_url, style_img_url_or_name = await _get_userdata_for_model_inputs(message.chat.id)
    if content_img_url is not None and style_img_url_or_name is not None and message.get_command() == '/start':
        # if both content and style are already given, then we are already processing some request
        return SendMessage(message.chat.id, "We have an active request already processing, please wait or execute /cancel to stop the request execution")
    # clean user data
    await _clean_userdata(message.chat.id)
    if message.get_command().lower() == '/start':
        return SendMessage(message.chat.id, "Please send me your image with content to stylize")
    elif content_img_url is None and style_img_url_or_name is None:
        return SendMessage(message.chat.id, "Nothing to cancel, please send me your image with content to stylize")
    else:
        if content_img_url is not None and style_img_url_or_name is not None:
            # this is necessary to prevent sending the resulting image for cancelled request, we just update the result counter
            _ = await _get_userdata_for_model_outputs(id, assume_userdata_exist=True, increment_counter=True)
        return SendMessage(message.chat.id, "The request is cancelled, to make a new request please send me your image with content to stylize")


@dp.message_handler(commands='help')
async def help_handler(message: types.Message):
    '''
    Handle help command
    '''

    return SendMessage(message.chat.id, 
        "Use <b>/start</b> or just send me your image with content to stylize to proceed with a new request, " +
        "use <b>/cancel</b> to cancel already processing one, " +
        f"{MESSAGE_TO_CONTACT_SUPPORT} before start using this bot to make sure all the necessary AWS resources are launched",
        parse_mode=types.ParseMode.HTML
    )


@dp.message_handler(content_types=[types.message.ContentType.PHOTO, types.message.ContentType.DOCUMENT])
async def photo_handler(message: types.Message):
    '''
    Process uploaded image, by checking whether each of content_img_url and style_img_url_or_name equals None or not we determine is it content image or style image
    '''

    content_img_url, style_img_url_or_name = await _get_userdata_for_model_inputs(message.chat.id)
    if content_img_url is not None and style_img_url_or_name is not None:
        return SendMessage(message.chat.id, "We have an active request already processing, please wait or execute /cancel to stop the request execution")
    file_id = None
    reply_str = None
    if message.document:
        # get file id for uncompressed image uploaded as a document
        if message.document.mime_type and message.document.mime_type.startswith('image/'):
            file_id = message.document.file_id
        else:
            reply_str = "This document is not an image"
    elif message.photo:
        # get file id for photo (i.e. compressed image), there are two images, take one with original size
        file_id = message.photo[-1].file_id
    else:
        reply_str = "Bad content"
    if file_id is not None:
        file_info = await bot.get_file(file_id)
        file_url = f'https://api.telegram.org/file/bot{API_TOKEN}/{file_info.file_path}'
        await _put_userdata_for_model_inputs(message.chat.id, 'content.txt' if content_img_url is None else 'style.txt', file_url, assume_userdata_exist=True)
        if content_img_url is None:
            content_img_url = file_url
            reply_str = "Successfully got the image with content to stylize"
        else: 
            style_img_url_or_name = file_url
            await bot.send_message(message.chat.id, "Successfully got the image with style to apply, processing the request...", reply_markup=types.ReplyKeyboardRemove())
            asyncio.create_task(_invoke_nst_endpoint(message.chat.id, content_img_url, style_img_url_or_name))
            return
    if reply_str is None:
        reply_str = "Bad content"
    if content_img_url is None:
        await bot.send_message(message.chat.id, f"{reply_str}, please send me your image with content to stylize")
    else:
        # content image is successfully processed, waiting for the style
        await bot.send_message(message.chat.id, f"{reply_str}, please send me your image with style to apply or choose the desired style name", reply_markup=get_stylechoice_markup())


@dp.message_handler(lambda message: message.text in list(CYCLEGAN_STYLENAME_TO_MODELNAME_DICT.keys()))
async def stylename_handler(message: types.Message):
    '''
    Process style name from the list of predefined styles
    '''

    content_img_url, style_img_url_or_name = await _get_userdata_for_model_inputs(message.chat.id)
    if content_img_url is not None and style_img_url_or_name is not None:
        return SendMessage(message.chat.id, "We have an active request already processing, please wait or execute /cancel to stop the request execution")
    if content_img_url is None:
        return SendMessage(message.chat.id, "Please send me your image with content to stylize")
    else:
        style_img_url_or_name = message.text
        await _put_userdata_for_model_inputs(message.chat.id, 'style.txt', style_img_url_or_name, assume_userdata_exist=True)
        await bot.send_message(message.chat.id, "Successfully got the style name to apply, processing the request...", reply_markup=types.ReplyKeyboardRemove())
        asyncio.create_task(_invoke_cyclegan_endpoint(message.chat.id, content_img_url, style_img_url_or_name))
        return

################################
# some handlers for wrong inputs
################################

async def _wronginput_handler(reason_str: str, message: types.Message):
    content_img_url, style_img_url_or_name = await _get_userdata_for_model_inputs(message.chat.id)
    if content_img_url is not None and style_img_url_or_name is not None:
        await bot.send_message(message.chat.id, "We have an active request already processing, please wait or execute /cancel to stop the request execution")
        return
    if content_img_url is None:
        await bot.send_message(message.chat.id, "Please send me your image with content to stylize")
    else:
        await bot.send_message(message.chat.id, f"{reason_str}, please send me your image with style to apply or choose the desired style name from the keyboard", reply_markup=get_stylechoice_markup())


@dp.message_handler(lambda message: message.text not in list(CYCLEGAN_STYLENAME_TO_MODELNAME_DICT.keys()))
async def wrongstylename_handler(message: types.Message):
   await _wronginput_handler("Bad style name", message)


@dp.message_handler(content_types=[types.message.ContentType.ANY])
async def wrongcontent_handler(message: types.Message):
    await _wronginput_handler("Bad content", message)


############################################################################################################
# auxiliary functions and handlers for invoking SageMaker endpoints and sending resulting images to Telegram
############################################################################################################

async def _sagemaker_invoke_endpoint(id, endpoint_name, input_params_dict, connect_timeout=60, read_timeout=60):
    '''
    Invoke SageMaker endpoint
    '''

    config = aiobotocore.config.AioConfig(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={
            'max_attempts': 0
        }
    )
    session = aiobotocore.get_session()
    async with session.create_client('sagemaker-runtime', **AWS_ACCESS_INFO_DICT, config=config) as sagemaker_client:
        result_img_url = await _get_userdata_for_model_outputs(id, assume_userdata_exist=True, increment_counter=True)
        input_params_dict['result_url'] = result_img_url
        sagemaker_response = await sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(input_params_dict),
            ContentType='application/json',
            Accept='application/json'
        )
        response_body = sagemaker_response['Body']
        async with response_body as stream:
            data = await stream.read()
            logging.info(json.loads(data.decode()))
    return result_img_url


async def _send_result_img(id, result_img_url, caption):
    '''
    Get resulting image from S3 and send it to Telegram
    '''

    current_result_img_url = await _get_userdata_for_model_outputs(id, assume_userdata_exist=True, increment_counter=False)
    if current_result_img_url == result_img_url:
        prefix = _get_prefix(id)
        async with aioboto3.resource("s3", **AWS_ACCESS_INFO_DICT) as s3:
            bucket_key = result_img_url[len("s3://") :]
            bucket_name, key = bucket_key.split("/", 1)
            result_s3_obj = await s3.Object(bucket_name, key)
            result_img_as_bytes = await _read_s3_obj_as_bytes(result_s3_obj)
        await bot.send_photo(id, result_img_as_bytes, caption=caption)                


async def _invoke_nst_endpoint(id, content_img_url, style_img_url):
    '''
    Process content image by NST
    '''

    # noinspection PyBroadException
    try:
        result_img_url = await _sagemaker_invoke_endpoint(
            id,
            'NeuralStyleTransfer',
            {'content_url': content_img_url, 'style_url': style_img_url, 'max_epochs': 14, 'n_steps_in_epoch': 10, 'lr': 0.01, 'patience': 5},
            connect_timeout=1200, read_timeout=1200
        )
        await _send_result_img(id, result_img_url, caption='stylized via NST')
    except Exception as e:
        logging.error(e)
        await bot.send_message(id, f"Request cannot be processed because the corresponding AWS resources are not launched, {MESSAGE_TO_CONTACT_SUPPORT}")
    await _clean_userdata(id)
    await bot.send_message(id, "For new request please send me your image with content to stylize")


async def _invoke_cyclegan_endpoint(id, content_img_url, style_name):
    '''
    Process content image by pretrained CycleGAN model determined by style name
    '''

    # noinspection PyBroadException
    try:
        model_name = CYCLEGAN_STYLENAME_TO_MODELNAME_DICT[style_name]
        result_img_url = await _sagemaker_invoke_endpoint(
            id,
            'CycleGANStyleTransfer',
            {'content_url': content_img_url, 'style_name': model_name},
            connect_timeout=60, read_timeout=60
        )
        await _send_result_img(id, result_img_url, caption=f'stylized via {style_name} CycleGAN')
    except Exception as e:
        logging.error(e)
        await bot.send_message(id, f"Request cannot be processed because the corresponding AWS resources are not launched, {MESSAGE_TO_CONTACT_SUPPORT}")
    await _clean_userdata(id)
    await bot.send_message(id, "For new request please send me your image with content to stylize")


async def on_startup(dp):
    logging.warning('Starting...')
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    logging.warning('Shutting down..')
    # Remove webhook (not acceptable in some cases)
    # await bot.delete_webhook()
    logging.warning('Bye!')


if __name__ == '__main__':
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=False,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
