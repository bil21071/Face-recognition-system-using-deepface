
import logging
from datetime import datetime
import os
import inspect  # Import inspect to get the function name

class CustomFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.team = os.getenv("TEAM")  # Hard-coded team name
        self.env = os.getenv("ENVI")  # Hard-coded environment
        self.ip = os.getenv("IP")  # Hard-coded IP address

    def format(self, record):
        record.team = self.team
        record.env = self.env
        record.ip = self.ip
        record.user_id = getattr(record, 'user_id', 'USER_ID')
        record.session_id = getattr(record, 'session_id', 'SESSION_ID')
        record.error_type = record.levelname  # Log level name (e.g., 'INFO', 'WARNING', 'ERROR')
        record.log_module = getattr(record, 'log_module', 'MODULE')
        record.call_userid = getattr(record, 'call_userid', 'CALL_USERID')
        record.device_name = getattr(record, 'device_name', 'DEVICE_NAME')
        record.fn_name = getattr(record, 'fn_name', 'FN_NAME')
        return super().format(record)

class Logs:
    def __init__(self) -> None:

        file_name = 'face_rec.log'
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(

            level=logging.INFO,
        )

        custom_formatter = CustomFormatter(
            fmt="%(team)s|%(env)s|%(asctime)s|%(ip)s|%(user_id)s|%(device_name)s|%(call_userid)s|%(session_id)s|%(error_type)s|%(log_module)s|%(fn_name)s|%(message)s",
            datefmt="%Y/%m/%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(custom_formatter)
        
        logger = logging.getLogger()
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        # Remove the default StreamHandler that basicConfig might add
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    def log(self, level, message: str, **kwargs) -> None:
        func_name = inspect.stack()[2].function
        extra = {
            'user_id': kwargs.get('user_id', 'USER_ID'),
            'session_id': kwargs.get('session_id', 'SESSION_ID'),
            'log_module': kwargs.get('log_module', 'MODULE'),
            'call_userid': kwargs.get('call_userid', 'CALL_USERID'),
            'device_name': kwargs.get('device_name', 'DEVICE_NAME'),
            'fn_name': func_name 
        }
        logging.getLogger().log(level, message, extra=extra)

    def error(self, message: str, **kwargs) -> None:
        self.log(logging.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.log(logging.WARNING, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self.log(logging.INFO, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self.log(logging.CRITICAL, message, **kwargs)

    def shutdown(self) -> None:
        logging.shutdown()


        # Method to disable logging
    def disable_logs(self):
        logging.disable(logging.CRITICAL + 1)

    # Method to enable logging
    def enable_logs(self):
        logging.disable(logging.NOTSET)



