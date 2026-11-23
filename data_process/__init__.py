"""离线数据准备、投影几何检查和ROI处理工具。"""
import os
import sys
from pathlib import Path

# 支持PyCharm直接使用Conda解释器，无需先在终端激活环境。
if os.name == 'nt':
    environment = Path(sys.executable).parent
    os.environ['PATH'] = os.pathsep.join([str(environment), str(environment/'Library'/'bin'),
                                        os.environ.get('PATH', '')])
