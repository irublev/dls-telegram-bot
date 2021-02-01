import asyncio
import logging
import os
from urllib.parse import urljoin
import aioboto3
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
}


def get_stylechoice_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True, one_time_keyboard=True)
    style_name_list = list(CYCLEGAN_STYLENAME_TO_MODELNAME_DICT.keys())
    markup.add(*style_name_list)
    return markup


async def _read_s3_obj_as_bytes(s3_obj):
    s3_obj_response = await s3_obj.get()
    s3_obj_body = s3_obj_response['Body']
    return await s3_obj_body.read()


async def _read_s3_obj_as_string(s3_obj):
   s3_obj_content = await _read_s3_obj_as_bytes(s3_obj)
   return s3_obj_content.decode("utf-8")


def _get_prefix(id):
   return f'userdata/{id}/'


async def _get_userdata_for_model_inputs(id, assume_userdata_exist=False):
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
                    result_couner = 0
                result_counter += 1
                await bucket.put_object(Key=f'{prefix}result_counter.txt', Body=str(result_counter).encode("utf-8"))
    if result_counter is not None:
        result_img_url = f's3://{AWS_DEFAULT_BUCKET}/{prefix}result{result_counter}.jpg'
    return result_img_url


async def _put_userdata_for_model_inputs(id, input_file_name, input_value, assume_userdata_exist=False):
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
            if obj.key != prefix:
                await obj.delete()



@dp.message_handler(commands=['start', 'cancel'])
async def start_or_cancel_handler(message: types.Message):
    content_img_url, style_img_url_or_name = await _get_userdata_for_model_inputs(message.chat.id)
    if content_img_url is not None and style_img_url_or_name is not None and message.get_command() == '/start':
        return SendMessage(message.chat.id, "We have an active request already processing, please wait or execute /cancel to stop the request execution")
    await _clean_userdata(message.chat.id)
    if message.get_command().lower() == '/start':
        return SendMessage(message.chat.id, "Please send me your image with content to stylize")
    elif content_img_url is None and style_img_url_or_name is None:
        return SendMessage(message.chat.id, "Nothing to cancel, please send me your image with content to stylize")
    else:
        return SendMessage(message.chat.id, "The request is cancelled, to make a new request please send me your image with content to stylize")


@dp.message_handler(commands='help')
async def help_handler(message: types.Message):
    return SendMessage(message.chat.id, "Use /start or just send me your image with content to stylize to proceed with a new request, use /cancel to cancel already processing one")


@dp.message_handler(content_types=[types.message.ContentType.PHOTO, types.message.ContentType.DOCUMENT])
async def photo_handler(message: types.Message):
    content_img_url, style_img_url_or_name = await _get_userdata_for_model_inputs(message.chat.id)
    if content_img_url is not None and style_img_url_or_name is not None:
        return SendMessage(message.chat.id, "We have an active request already processing, please wait or execute /cancel to stop the request execution")
    file_id = None
    reply_str = None
    if message.document:
        if message.document.mime_type and message.document.mime_type.startswith('image/'):
            file_id = message.document.file_id
        else:
            reply_str = "This document is not an image"
    elif message.photo:
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
        await bot.send_message(message.chat.id, f"{reply_str}, please send me your image with style to apply or choose the desired style name", reply_markup=get_stylechoice_markup())


@dp.message_handler(lambda message: message.text in list(CYCLEGAN_STYLENAME_TO_MODELNAME_DICT.keys()))
async def stylename_handler(message: types.Message):
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


async def _invoke_nst_endpoint(id, content_img_url, style_img_url):
    # noinspection PyBroadException
    try:
        config = aiobotocore.config.Config(
            read_timeout=1200,
            retries={
                'max_attempts': 0
            }
        )
        #sagemaker_runtime_client = boto3.client('sagemaker-runtime', config=config)
        #sagemaker_client = Session(sagemaker_runtime_client=sagemaker_runtime_client)        
        session = aiobotocore.get_session()
        async with session.create_client('sagemaker-runtime', config=config) as sagemaker_client:
            result_img_url = await _get_userdata_for_model_outputs(assume_userdata_exist=True, increment_counter=True)
            _ = await client.invoke_endpoint(
                EndpointName='NeuralStyleTransfer',
                Body=json.dumps({'content_url': content_img_url, 'style_url': style_img_url, 'result_url': result_img_url, 'max_epochs': 10, 'n_steps_in_epoch': 20}),
                ContentType='application/json',
                Accept='application/json'
            )
        current_result_img_url = await _get_userdata_for_model_outputs(assume_userdata_exist=True, increment_counter=False)
        if current_result_img_url == result_img_url:
            prefix = _get_prefix(id)
            async with aioboto3.resource("s3", **AWS_ACCESS_INFO_DICT) as s3:
                bucket = await s3.Bucket(AWS_DEFAULT_BUCKET)
                bucket_key = result_img_url[len("s3://") :]
                bucket_name, key = bucket_key.split("/", 1)
                result_s3_obj = await bucket.get_object(Key=key)
                result_img_as_bytes = _read_s3_obj_as_bytes(result_s3_obj)
            await bot.send_photo(id, result_img_as_bytes, caption='stylized via NST')                
    except Exception as e:
        logging.error(e)
        await _clean_userdata(id)
        await bot.send_message(id, f"Requests cannot be processed because the corresponding AWS resources are not launched, {MESSAGE_TO_CONTACT_SUPPORT}")


async def _invoke_cyclegan_endpoint(id, content_img_url, style_name):
    # noinspection PyBroadException
    try:
        # invoke_endpoint
        await asyncio.sleep(45)
        raise Exception("Not yet implemented!")
    except Exception as e:
        logging.error(e)
        await _clean_userdata(id)
        await bot.send_message(id, f"Requests cannot be processed because the corresponding AWS resources are not launched, {MESSAGE_TO_CONTACT_SUPPORT}")


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
