import logging
from colorama import Fore, Style, init


class colorful_logger:

    def __init__(self, name, logfile=None):
        self.name = name
        init(autoreset=True)

        self.logger = logging.getLogger(name)

<<<<<<< HEAD
        # 不带颜色的日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

        # 文件日志处理器 (不带颜色)
=======
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

>>>>>>> dev
        if self.name == 'Train' and logfile:
            filehandler = logging.FileHandler(logfile)
            filehandler.setLevel(logging.INFO)
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)

        self.logger.setLevel(logging.INFO)

        if self.name != 'Train':
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_with_color(self, message=None, color=Fore.WHITE):

        if self.name == 'Evaluate':
            color = Fore.CYAN
        elif self.name == 'Inference':
            color = Fore.MAGENTA

        if self.name == 'Train':
            colored_message = message
        else:
            colored_message = f"{color}{message}{Style.RESET_ALL}"

<<<<<<< HEAD
        # 控制台输出带颜色的日志
=======
>>>>>>> dev
        self.logger.info(colored_message)


# Usage-------------------------------------------------------------
def main():
    test = colorful_logger('Inference')

    test.log_with_color('This is a debug message')
    test.log_with_color('This is an info message')
    test.log_with_color('This is a critical message')


if __name__ == '__main__':
    main()