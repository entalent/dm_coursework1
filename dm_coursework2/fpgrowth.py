# -*- coding: utf-8 -*-
# https://www.ibm.com/developerworks/cn/

def loadDataSet():
    dataSet = [['bread', 'milk', 'vegetable', 'fruit', 'eggs'],
               ['noodle', 'beef', 'pork', 'water', 'socks', 'gloves', 'shoes', 'rice'],
               ['socks', 'gloves'],
               ['bread', 'milk', 'shoes', 'socks', 'eggs'],
               ['socks', 'shoes', 'sweater', 'cap', 'milk', 'vegetable', 'gloves'],
               ['eggs', 'bread', 'milk', 'fish', 'crab', 'shrimp', 'rice']]
    return dataSet

def to_frozen(dataSet):
    frozenDataSet = {}
    for elem in dataSet:
        frozenDataSet[frozenset(elem)] = 1
    return frozenDataSet

class TreeNode:
    def __init__(self, nodeName, count, nodeParent):
        self.nodeName = nodeName
        self.count = count
        self.nodeParent = nodeParent
        self.nextSimilarItem = None
        self.children = {}

    def increaseC(self, count):
        self.count += count

    def disp(self):
        print(self.nodeName, self.count, self.nextSimilarItem)
        for c in self.children.values():
            c.disp()

def create_fptree(data, minSupport):
    '''
    创建FP树
    :param data:
    :param minSupport:
    :return:
    '''
    # 第一次扫描，获取频繁项
    head_point_table = {}
    for items in data:
        for item in items:
            head_point_table[item] = head_point_table.get(item, 0) + data[items]
    head_point_table = {k:v for k,v in head_point_table.items() if v >= minSupport}
    frequentItems = set(head_point_table.keys())
    if len(frequentItems) == 0: return None, None

    for k in head_point_table:
        head_point_table[k] = [head_point_table[k], None]
    fptree = TreeNode("null", 1, None)
    #scan dataset at the second time, filter out items for each record
    # 第二次扫描，更新项头表以及创建FP树
    for items,count in data.items():
        frequentItemsInRecord = {}
        for item in items:
            if item in frequentItems:
                frequentItemsInRecord[item] = head_point_table[item][0]
        if len(frequentItemsInRecord) > 0:
            orderedFrequentItems = [v[0] for v in sorted(frequentItemsInRecord.items(), key=lambda v:v[1], reverse = True)]
            updateFPTree(fptree, orderedFrequentItems, head_point_table, count)

    return fptree, head_point_table

def updateFPTree(fptree, orderedFrequentItems, headPointTable, count):
    # 向FP树中添加一条记录
    #handle the first item
    if orderedFrequentItems[0] in fptree.children:
        fptree.children[orderedFrequentItems[0]].increaseC(count)
    else:
        fptree.children[orderedFrequentItems[0]] = TreeNode(orderedFrequentItems[0], count, fptree)

        #update headPointTable
        if headPointTable[orderedFrequentItems[0]][1] == None:
            headPointTable[orderedFrequentItems[0]][1] = fptree.children[orderedFrequentItems[0]]
        else:
            updateHeadPointTable(headPointTable[orderedFrequentItems[0]][1], fptree.children[orderedFrequentItems[0]])
    #handle other items except the first item
    if(len(orderedFrequentItems) > 1):
        updateFPTree(fptree.children[orderedFrequentItems[0]], orderedFrequentItems[1::], headPointTable, count)

def updateHeadPointTable(headPointBeginNode, targetNode):
    # 更新项头表
    while(headPointBeginNode.nextSimilarItem != None):
        headPointBeginNode = headPointBeginNode.nextSimilarItem
    headPointBeginNode.nextSimilarItem = targetNode

def mine_fptree(headPointTable, prefix, frequentPatterns, minSupport):
    # 挖掘频繁项集
    # 获得频繁项的前缀路径，对条件FP树中的每个频繁项获得前缀路径并构建新的条件FP树
    headPointItems = [v[0] for v in sorted(headPointTable.items(), key = lambda v:v[1][0])]
    if(len(headPointItems) == 0): return
    for headPointItem in headPointItems:
        newPrefix = prefix.copy()
        newPrefix.add(headPointItem)
        support = headPointTable[headPointItem][0]
        frequentPatterns[frozenset(newPrefix)] = support

        # 频繁项的前缀路径
        prefixPath = getPrefixPath(headPointTable, headPointItem)
        if(prefixPath != {}):
            # 构造条件FP树
            conditionalFPtree, conditionalHeadPointTable = create_fptree(prefixPath, minSupport)
            if conditionalHeadPointTable != None:
                mine_fptree(conditionalHeadPointTable, newPrefix, frequentPatterns, minSupport)

def getPrefixPath(headPointTable, headPointItem):
    # 获取频繁项的前缀路径
    prefixPath = {}
    beginNode = headPointTable[headPointItem][1]
    prefixs = ascendTree(beginNode)
    if((prefixs != [])):
        prefixPath[frozenset(prefixs)] = beginNode.count

    while(beginNode.nextSimilarItem != None):
        beginNode = beginNode.nextSimilarItem
        prefixs = ascendTree(beginNode)
        if (prefixs != []):
            prefixPath[frozenset(prefixs)] = beginNode.count
    return prefixPath

def ascendTree(treeNode):
    prefixs = []
    while((treeNode.nodeParent != None) and (treeNode.nodeParent.nodeName != 'null')):
        treeNode = treeNode.nodeParent
        prefixs.append(treeNode.nodeName)
    return prefixs

rules = []

def rulesGenerator(frequentPatterns, minConf, total_items):
    global rules
    for (i, frequentset) in enumerate(frequentPatterns):
        if i % 100 == 0:
            print('{} of {} {}'.format(i, len(frequentPatterns), len(rules)))
        if(len(frequentset) > 1):
            getRules(frequentset, frequentset, frequentPatterns, minConf, total_items)

def removeStr(set, str):
    tempSet = []
    for elem in set:
        if(elem != str):
            tempSet.append(elem)
    tempFrozenSet = frozenset(tempSet)
    return tempFrozenSet


def getRules(frequentset, currentset, frequentPatterns, minConf, total_items):
    global rules
    for frequentElem in currentset:
        subSet = removeStr(currentset, frequentElem)
        a = frequentPatterns[frequentset]
        if subSet not in frequentPatterns:
            return
        b = frequentPatterns[subSet]

        # frequentset: X U Y
        # subset: X
        confidence = float(a) / float(b)
        if frequentset - subSet not in frequentPatterns:
            return
        if (confidence >= minConf):
            flag = False
            for rule in rules:
                if(rule[0] == subSet and rule[1] == frequentset - subSet):
                    flag = True
            support = frequentPatterns[frequentset] / total_items
            lift = confidence / (frequentPatterns[frequentset - subSet] / total_items)
            if(flag == False):
                rules.append((subSet, frequentset - subSet, support, confidence, lift))

            if(len(subSet) >= 2):
                getRules(frequentset, subSet, frequentPatterns, minConf, total_items)

if __name__=='__main__':
    print("fptree:")
    dataSet = loadDataSet()
    frozenDataSet = to_frozen(dataSet)
    minSupport = 3
    fptree, headPointTable = create_fptree(frozenDataSet, minSupport)
    # fptree.disp()
    frequentPatterns = {}
    prefix = set([])
    mine_fptree(headPointTable, prefix, frequentPatterns, minSupport)
    print("frequent patterns:")
    for i in frequentPatterns.items():
        print(i)
    minConf = 0.6

    total = len(dataSet)
    rulesGenerator(frequentPatterns, minConf, float(total))
    print("association rules:")
    for i in rules:
        x, y = i[0], i[1]
        support = i[2]
        confidence = i[3]
        lift = i[4]
        print(x, y, support, confidence, lift)