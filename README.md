# dm_coursework1
data mining coursework 1
赵文天 2120171105

## 环境
- Windows 10 x64
- Python 3.5
- numpy, matplotlib, scipy, pandas

## 项目结构
results 所有结果
    results/BuildingPermits   Building Permits原始数据集的结果
        results/BuildingPermits/plots   所有数值属性的图表，每个文件中包含一个属性的直方图，Q-Q图，盒图
        results/BuildingPermits/digest_BuildingPermits_nominal.json   所有标称属性的频数的统计，json格式
        results/BuildingPermits/digest_BuildingPermits_nominal.json   所有数值属性的摘要，json格式
    results/NFL NFL play by play原始数据集的结果
        results/NFL/plots   所有数值属性的图表，每个文件中包含一个属性的直方图，Q-Q图，盒图
        results/NFL/digest_NFL_nominal.json   所有标称属性的频数的统计，json格式
        results/NFL/digest_NFL_nominal.json   所有数值属性的摘要，json格式
    results/imputed   缺失值填补之后的数据摘要以及图表
        results/imputed/column    通过属性之间的相关关系填补缺失值后的结果
        results/imputed/knn       通过数据对象之间的相似性填补缺失值后的结果
        results/imputed/mode      用最高频率值填补缺失值后的结果
1.csv   对NFL play by play数据集中的数值属性的标注
2.csv   对Building Permits数据集中的数值属性的标注
main.py   对原始数据集中的数值属性进行摘要，画出图表
main1.py  用不同方法对缺失值进行填补，对填补后的数据集进行摘要，画出图表
