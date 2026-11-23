# 数据处理脚本

在PyCharm选择pytorch_3.7解释器，直接运行需要的脚本即可。

| 文件 | 作用 | 当前是否需要重新运行 |
|---|---|---|
| prepare_search_rois.py | 复用计划标签，检查±10mm固定搜索ROI，输出说明和一张示例图 | 本次已完成 |
| search_roi.py | ROI边界计算、裁剪及补零工具，由Dataset调用 | 无需单独运行 |
| prepare_center_labels.py | 旧单患者的三维中心投影标签准备，以及公用标签定义/哈希工具 | 现有标签可复用 |
| calibrate_drr_demo.py | 实际Demo几何标定及多患者中心标签生成 | 已完成，不必重跑Demo |
| check_geometry.py | 旧单患者TIGRE投影几何检查 | 历史工具，不必重跑 |
| make_nodule_mask_drr.py | 肿瘤投影图阈值二值化 | 现有Mask可复用 |
| state_usage.py | 按原患者划分和±10mm规则标注状态用途，供训练/评估调用 | train.py自动调用，无需另存预处理副本 |
| generate_projection_matrices.py | 实际Demo标定，独立体模与全部状态投影核对；只保存每患者矩阵与质检结论 | 新患者数据使用此入口 |
| generate_center_labels.py | 使用已验证矩阵，计算三维Mask物理质心及kv1/kv2投影标签 | 矩阵生成后运行；不调用Demo/DLL |

文件移动后，项目根目录仍为本目录的上一级，数据和outputs路径未改变。
train.py、test.py、run_localization.py、network.py及Dataset在根目录，保持训练预测入口清楚。
preprocessing保留原DRR生成实现，避免改变已经验证的原投影程序。

## 新数据：只生成投影矩阵

在Windows、可运行实际DRRDemo和CUDA的环境中，使用pytorch_3.7直接运行
generate_projection_matrices.py。默认扫描Data/New_data的全部患者，支持Origin与original。
后续只需修改该脚本顶部DATA_ROOT为400多例患者所在根目录；每个患者直接位于其下。
DEMO_ROOT指向实际生成这些DRR的Demo目录，必须包含DRRDemo.exe、DRR.dll、cudart64_102.dll及paraset.ini。

输出自动放在outputs/geometry_数据目录名：
- patients/患者ID/geometry.json：kv1和kv2的3×4矩阵、坐标约定、计划信息和少量核验数字。
- 质检结论.md：每位患者是否通过及误差汇总。

数据、程序和设置不变时会核对指纹后复用结果；已有结果来源变化时拒绝覆盖，请指定新的--out。
生成过程中临时副本均位于本次输出目录，结束后清理；原数据和原Demo配置不会被改写。
也支持命令行--root、--demo、--out。不生成网络训练标签，不重新训练；本步骤不依赖训练/测试划分。
后续训练可以在Linux服务器读取矩阵JSON，但此Demo标定入口依赖Windows程序，不能直接在Linux执行。

## 新数据：生成中心标签

完成矩阵标定后，在pytorch_3.7中直接运行generate_center_labels.py。
默认读取Data/New_data，结果写入已有outputs/geometry_New_data，不重新投影CT或Mask。
后续修改脚本顶部DATA_ROOT为患者根目录；默认对应outputs/geometry_数据目录名。
若矩阵保存在其他位置，用--geometry-root指定前一步输出目录。

新增输出：
- patients/患者ID/projected_centers.json：Origin及全部phase的三维中心和两视角二维标签，供监督和评估使用。
- patients/患者ID/original_labels.json：仅保留计划状态，供后续预测使用。
- centers.csv：每状态一行的中心坐标总表，包含相对计划中心位移及±10mm范围标记。
- 中心标签质检结论.md：每患者是否通过、状态与标签数量。

三维中心为0/1 Mask内所有前景体素中心的平均物理位置，使用患者LPS坐标、毫米单位。
二维标签为该三维中心经对应矩阵的投影，使用完整DRR中的零起始[u,v]像素坐标，保留小数。
JSON中计划状态统一记作original，source_folder记录实际Origin目录，不改动原数据。
±10mm标记按LPS各轴位移判断；不删除状态、不重新划分患者。

脚本检查来源指纹、Mask与标定网格、独立物理质心及投影坐标，检查不通过的患者不会生成新标签。
数据、矩阵和标签算法不变时复用已生成标签。原geometry.json、原数据和历史结果保持不变。
只需NumPy、SimpleITK及上述已有Python工具；本步骤不调用Demo、DLL、TIGRE或神经网络。
为核对来源，需能读取前一步标定使用的完整患者目录，不能只复制Mask。
本入口只准备标签；训练入口对新目录结构的适配及400例数据划分接入是后续工作。
