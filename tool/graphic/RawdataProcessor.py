class RawdataProcessor:
    """
    统一用来处理原始数据的类
    """
    def __init__(self, data_path: str, drone_name: str):
        self.data_path = data_path