"""训练/测试共用地址；上传服务器后修改这里，或使用命令行覆盖。"""
import os
import sys
from pathlib import Path


# 先初始化运行环境；其他脚本在导入numpy/torch/SimpleITK前导入本模块。
if os.name == 'nt':
    environment = Path(sys.executable).parent
    os.environ['PATH'] = os.pathsep.join([
        str(environment), str(environment/'Library'/'bin'), os.environ.get('PATH', '')])
os.environ['MPLBACKEND'] = 'Agg'

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DATA_ROOT = Path(r'D:\BotView\02_code\01_DataSet\trainDataSet1')
TRAIN_PREPARATION_ROOT = Path(r'D:\BotView\02_code\01_DataSet\trainDataSetLabel1')
TRAIN_OUTPUT_ROOT = PROJECT_ROOT/'outputs'/'manifest_training'

# 测试原始图像与预处理结果分开存放；先在预处理目录生成manifest_test/dataset.json。
TEST_DATA_ROOT = Path(r'D:\BotView\02_code\01_DataSet\trainDataSet1')
TEST_PREPARATION_ROOT = Path(r'D:\BotView\02_code\01_DataSet\trainDataSetLabel1')
TEST_CHECKPOINT = None             # 例如Path('/server/runs/某次训练/last.pt')
TEST_OUTPUT_ROOT = PROJECT_ROOT/'outputs'/'manifest_evaluation'
