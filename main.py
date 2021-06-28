import constants
import io
import os
import telebot
# import pyTelegramBotAPI
from model import get_cnn, style_transfer, get_great_cnn


users = dict()
cnn, dtype = get_cnn()
great_cnn = get_great_cnn()

STYLES_FOLDER = ""


def get_str_time(sec):
    h_int = int(sec / 3600)
    h = f"{str(h_int)} ч. " if h_int > 0 else ""
    min_int = int((sec % 3600) / 60)
    m = f"{str(min_int)} мин. "
    s = f"{str(int(sec % 60))} сек."
    return h + m + s


def get_string_progress(num, used, left):
    green = int(num / 10)
    return f"|> Идет обработка...\n" \
           f"|> В ожидании: {get_str_time(used)}\n" \
           f"|> Осталось ждать: {get_str_time(left)}\n" \
           f"Завершено: {num}%\n[" + "✅" * green + "❌" * (10 - green) + "]"


def get_key_bot_type_str(user_id):
    return str(user_id) + "__bot_type"


def get_bot_type(user_id):
    try:
        return users[get_key_bot_type_str(user_id)]
    except:
        users[get_key_bot_type_str(user_id)] = constants.GREAT_CONST
        return constants.GREAT_CONST


def main():

    bot = telebot.TeleBot(constants.token)

    def edit_message(num, used, left,  message, mes_id):
        try:
            bot.edit_message_text(get_string_progress(num, used, left), message.chat.id, mes_id)
        except:
            pass

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        m = bot.reply_to(message, constants.start_message)
        # print("|> Message start:", m)

    @bot.message_handler(commands=['help'])
    def send_welcome(message):
        bot.reply_to(message, constants.help_message)

    @bot.message_handler(commands=['set_great_model'])
    def send_welcome(message):
        users[get_key_bot_type_str(message.from_user.id)] = constants.GREAT_CONST
        bot.send_message(message.chat.id, constants.change_settings)

    @bot.message_handler(commands=['set_usual_model'])
    def send_welcome(message):
        users[get_key_bot_type_str(message.from_user.id)] = constants.USUAL_CONST
        bot.send_message(message.chat.id, constants.change_settings)

    @bot.message_handler(content_types=['text'])
    def message_post(message):
        bot.send_message(message.chat.id, constants.text_message)

    @bot.message_handler(content_types=['photo'])
    def message_post(message):
        if message.chat.type == 'private':
            user_id = message.from_user.id
            f_id = message.photo[-1].file_id
            file_info = bot.get_file(f_id)
            down_file = bot.download_file(file_info.file_path)
            if user_id in users:
                file_image = users[user_id]
                users.pop(user_id)
                style_filename = "style__" + str(user_id) + '.jpg'
                with open(style_filename, 'wb') as file:
                    file.write(down_file)
                message = bot.send_message(message.chat.id, get_string_progress(0, 60 * 20, 0))
                mes_id = message.message_id

                params1 = {
                    'cnn': great_cnn if get_bot_type(user_id) else cnn,
                    'dtype': dtype,
                    'content_image': file_image,
                    'style_image': style_filename,
                    'image_size': 512,
                    'style_size': 512,
                    'content_layer': 3,
                    'content_weight': 5e-2,
                    'style_layers': (1, 4, 6, 7),
                    'style_weights': (20000, 500, 12, 1),
                    'tv_weight': 5e-2,
                    'progress': lambda num, used, left: edit_message(num, used, left, message, mes_id)
                }

                new_img = style_transfer(**params1)
                img_byte_arr = io.BytesIO()
                new_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                bot.send_message(message.chat.id, "Изображение готово!")
                bot.send_photo(message.chat.id, img_byte_arr)

                os.remove(style_filename)
                os.remove(file_image)

            else:
                filename = "image_" + str(user_id) + '.jpg'
                users[user_id] = filename
                with open(filename, 'wb') as file:
                    file.write(down_file)
                bot.send_message(message.chat.id, "Отлично! Теперь жду изображение со стилем")

    bot.polling(none_stop=True)


if __name__ == '__main__':
    main()
