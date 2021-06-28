import constants
import io
import os
import telebot
# import pyTelegramBotAPI
from model import get_cnn, style_transfer

count = 0

users = dict()
cnn, dtype = get_cnn()

STYLES_FOLDER = ""


def get_string_progress(num):
    green = int(num / 10)
    return f"|> Идет обработка...\n" \
           f"Завершено: {num}%\n[" + "✅" * green + "❌" * (10 - green) + "]"


def main():

    bot = telebot.TeleBot(constants.token)

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        m = bot.reply_to(message, constants.start_message)
        print("|> Message start:", m)
        # mes_id = m.message_id
        # bot.edit_message_text("111111", message.chat.id, mes_id, )

    @bot.message_handler(commands=['help'])
    def send_welcome(message):
        bot.reply_to(message, constants.help_message)


    @bot.message_handler(content_types=['photo'])
    def message_post(message):
        if message.chat.type == 'private':
            user_id = message.from_user.id
            f_id = message.photo[-1].file_id
            file_info = bot.get_file(f_id)
            down_file = bot.download_file(file_info.file_path)
            global count
            if user_id in users:
                file_image = users[user_id]
                users.pop(user_id)
                filename = "style_" + str(user_id) + '.jpg'
                with open(filename, 'wb') as file:
                    file.write(down_file)
                message = bot.send_message(message.chat.id, get_string_progress(0))
                mes_id = message.message_id
                # bot.edit_message_text("111111", message.chat.id, mes_id, )

                params1 = {
                    'cnn': cnn,
                    'dtype': dtype,
                    'content_image': file_image,
                    'style_image': filename,
                    'image_size': 192,
                    'style_size': 512,
                    'content_layer': 3,
                    # 'content_layer' : 1,
                    'content_weight': 5e-2,
                    'style_layers': (1, 4, 6, 7),
                    'style_weights': (20000, 500, 12, 1),
                    # 'style_layers' : (1, 4),
                    # 'style_weights' : (20000, 500),
                    'tv_weight': 5e-2,
                    'progress': lambda x: bot.edit_message_text(get_string_progress(x), message.chat.id, mes_id)
                }

                new_img = style_transfer(**params1)
                os.remove(filename)
                os.remove(file_image)
                img_byte_arr = io.BytesIO()
                new_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                bot.send_message(message.chat.id, "Вот твоя картинка")
                bot.send_photo(message.chat.id, img_byte_arr)
            else:
                filename = "image_" + str(user_id) + '.jpg'
                users[user_id] = filename
                with open(filename, 'wb') as file:
                    file.write(down_file)
                #print(down_file)
                bot.send_message(message.chat.id, "Теперь загрузи изображение со стилями")
            count += 1
    bot.polling(none_stop=True)


if __name__ == '__main__':
    main()
