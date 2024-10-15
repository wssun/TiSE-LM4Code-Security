import json
import logging

# 设置日志配置
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_jsonl_to_dict(file_path):
    """读取JSONL文件到字典。"""
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.update(json_data)
    return data

def compare_dicts(dict1, dict2):
    # 找到两个字典共有的键
    common_keys = dict1.keys() & dict2.keys()

    # 创建一个字典来保存共有键及其值的和
    common_keys_values_sum = {key: dict1[key] + dict2[key] for key in common_keys}
    # 根据值的和排序
    sorted_common_keys = sorted(common_keys_values_sum.items(), key=lambda item: item[1], reverse=True)

    # 日志记录排序后的共有键及其总和
    logging.info("共有的键及其在两个字典中的值之和 (按总和排序):")
    for key, value_sum in sorted_common_keys:
        logging.info(f"{key}: dict1={dict1[key]}, dict2={dict2[key]}, sum={value_sum}")


    # 找到只在 dict1 中的键及其值，并按值大小排序，显示前10个
    unique_keys_dict1 = {k: dict1[k] for k in dict1.keys() - dict2.keys()}
    sorted_unique_keys_dict1 = sorted(unique_keys_dict1.items(), key=lambda item: item[1], reverse=True)
    logging.info("只在第一个字典中的键及其值 (前10，按值排序):")
    for key, value in sorted_unique_keys_dict1:
        logging.info(f"{key}: {value}")

    # 找到只在 dict2 中的键及其值，并按值大小排序，显示前10个
    unique_keys_dict2 = {k: dict2[k] for k in dict2.keys() - dict1.keys()}
    sorted_unique_keys_dict2 = sorted(unique_keys_dict2.items(), key=lambda item: item[1], reverse=True)
    logging.info("只在第二个字典中的键及其值 (前10，按值排序):")
    for key, value in sorted_unique_keys_dict2:
        logging.info(f"{key}: {value}")

    # 找到两个字典中值最大的20个键及其值
    top_n_keys_dict1 = sorted(dict1.items(), key=lambda item: item[1], reverse=True)[:10]
    top_n_keys_dict2 = sorted(dict2.items(), key=lambda item: item[1], reverse=True)[:10]

    logging.info("第一个字典中值最大的10个键和值:")
    for key, value in top_n_keys_dict1:
        logging.info(f"{key}: {value}")

    logging.info("第二个字典中值最大的10个键和值:")
    for key, value in top_n_keys_dict2:
        logging.info(f"{key}: {value}")

# JSONL文件路径
file_path0 = './dataset/label_0.jsonl'
file_path1 = './dataset/label_1.jsonl'

# 读取字典
dict0 = read_jsonl_to_dict(file_path0)
dict1 = read_jsonl_to_dict(file_path1)


# dict1 = {'a': 1, 'b': 2, 'c': 3}
# dict2 = {'b': 4, 'c': 5, 'd': 6}

# 比较字典
compare_dicts(dict0, dict1)
