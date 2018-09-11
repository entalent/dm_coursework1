# dm_coursework1
data mining coursework 1
赵文天 2120171105

## 环境
- Windows 10 x64
- Python 3.5
- numpy, matplotlib, scipy, pandas

## 项目结构
results 所有结果</br>
    results/BuildingPermits   Building Permits原始数据集的结果</br>
        results/BuildingPermits/plots  所有数值属性的图表，每个文件中包含一个属性的直方图，Q-Q图，盒图 </br>
        results/BuildingPermits/digest_BuildingPermits_nominal.json   所有标称属性的频数的统计，json格式 </br>
        results/BuildingPermits/digest_BuildingPermits_nominal.json   所有数值属性的摘要，json格式 </br>
    results/NFL NFL play by play原始数据集的结果 </br>
        results/NFL/plots   所有数值属性的图表，每个文件中包含一个属性的直方图，Q-Q图，盒图 </br>
        results/NFL/digest_NFL_nominal.json   所有标称属性的频数的统计，json格式 </br>
        results/NFL/digest_NFL_nominal.json   所有数值属性的摘要，json格式 </br>
    results/imputed   缺失值填补之后的数据摘要以及图表 </br>
        results/imputed/column    通过属性之间的相关关系填补缺失值后的结果 </br>
        results/imputed/knn       通过数据对象之间的相似性填补缺失值后的结果 </br>
        results/imputed/mode      用最高频率值填补缺失值后的结果 </br>
1.csv   对NFL play by play数据集中的数值属性的标注 </br>
2.csv   对Building Permits数据集中的数值属性的标注 </br>
main.py   对原始数据集中的数值属性进行摘要，画出图表 </br>
main1.py  用不同方法对缺失值进行填补，对填补后的数据集进行摘要，画出图表 </br>
