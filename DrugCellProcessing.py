import csv
import math

import pubchempy as pcp
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from smiles2graph import *
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# 第一步  获取反应表中存在的药物和细胞系名称
def getDrugAndCellLineNames():
    data = pd.read_csv('../RawData/drug_response_data.csv')
    # 获取所有反应数据里面4存在的细胞系和药物名称
    drugNameList = pd.DataFrame(set(data['drug_name']))
    cellLineNameList = pd.DataFrame(set(data['ccle_name']))
    drugNameList.to_csv('../RawData/drug/drugNameList.csv', index=False)
    cellLineNameList.to_csv('../RawData/Cell/cellLineNameList.csv', index=False)


def processingResponseData():
    data = pd.read_csv('../RawData/drug_response_data.csv')
    endData = pd.read_csv('../RawData/drug_response_data.csv')
    cellLineNames = pd.read_csv('../RawData/Cell/cellLineName402.csv')
    for i in range(data.shape[0]):
        index = data.shape[0] - i - 1
        if data.iloc[index, 0] not in cellLineNames['cellLineName'].tolist():
            endData.drop(index=index, axis=0, inplace=True)
            print('删除{}号反应记录--'.format(index))
    endData.to_csv('../RawData/drug_response_data_{}.csv'.format(endData.shape[0]), index=False)


def BoxResponseIC50():
    data = pd.read_csv('../RawData/drug_response_data.csv')
    data = data.sort_values(by=["ic50"])
    Q1_index = math.ceil(len(data) / 4)
    Q2_index = math.ceil(len(data) / 4 * 2)
    Q3_index = math.ceil(len(data) / 4 * 3)
    Q1 = data.values[Q1_index][6]
    Q2 = data.values[Q2_index][6]
    Q3 = data.values[Q3_index][6]
    print("Q1,Q2,Q3", Q1, Q2, Q3)
    IQR = Q3 - Q1
    LOW = Q1 - 1.5 * IQR
    HIGH = Q3 + 1.5 * IQR
    print("IQR, LOW, HIGH", IQR, LOW, HIGH)
    drug_cell_label = data.loc[data["ic50"] >= LOW]
    drug_cell_label = drug_cell_label.loc[drug_cell_label["ic50"] <= HIGH].values
    print("after box-polt", len(drug_cell_label))

    min_max = MinMaxScaler()
    temp = drug_cell_label[:, 6].reshape(-1, 1)
    temp = min_max.fit_transform(temp)
    drug_cell_label[:, 6] = temp.reshape(-1)
    drug_cell_label = pd.DataFrame(drug_cell_label)
    drug_cell_label.columns = ['ccle_name', 'drug_name', 'drug_smiles', 'pubchem_cid', 'auc', 'ec50', 'ic50',
                               'log-ic50', 'log-normal']
    drug_cell_label.to_csv('../RawData/drug_response_data_{}.csv'.format(len(drug_cell_label)), index=False)


def drugResponseDataToNumber():
    data = pd.read_csv('../RawData/drug_response_data.csv')
    drugNameList = pd.read_csv('../RawData/drug/drugNameInfo1353.csv')['drug_name'].tolist()
    cellLineNameList = pd.read_csv('../RawData/Cell/cellLineName402.csv', index_col=0)['cellLineName'].tolist()
    responseDrugName = data['drug_name'].tolist()
    responseCellLineName = data['ccle_name'].tolist()
    ic50 = data['ic50'].tolist()
    responseData = []

    for i in range(len(responseDrugName)):
        print('处理第{}条数据---'.format(i))
        currentRecord = []
        drugIndex = drugNameList.index(responseDrugName[i])
        cellLineIndex = cellLineNameList.index(responseCellLineName[i])
        currentRecord.append(drugIndex)
        currentRecord.append(cellLineIndex)
        currentRecord.append(ic50[i])
        responseData.append(currentRecord)
    d = pd.DataFrame(responseData)
    d.columns = ['drug', 'cellLine', 'ic50']
    d.to_csv('../RawData/drug_response_data_number.csv', index=False)


# 获取有四种细胞系特征数据的细胞系名称
def GetCellLineNames():
    cellLineNameList = pd.read_csv('../RawData/Cell/cellLineNameList.csv')
    geneExp = pd.read_csv('../RawData/Cell/Gene Expression.csv', index_col=0)
    CNV = pd.read_csv('../RawData/Cell/Copy Number Alterations.csv', index_col=0)
    miRNA = pd.read_csv('../RawData/Cell/miRNA.csv', index_col=0)
    Methylation = pd.read_csv('../RawData/Cell/Methylation.csv', index_col=0)

    # 获取行名，即细胞系名
    geneExpIndex = geneExp.index
    CNVIndex = CNV.index
    miRNAIndex = miRNA.index
    MethylationIndex = Methylation.index

    # 定义能够搜索到相应特征的细胞系名称
    geneExpIndexList = []
    CNVIndexList = []
    miRNAIndexList = []
    MethylationIndexList = []

    # 遍历搜索有对应特征的细胞系名称
    for c in cellLineNameList['0']:
        if c in geneExpIndex:
            geneExpIndexList.append(c)

    for c in cellLineNameList['0']:
        if c in CNVIndex:
            CNVIndexList.append(c)

    for c in cellLineNameList['0']:
        if c in miRNAIndex:
            miRNAIndexList.append(c)

    for c in cellLineNameList['0']:
        if c in MethylationIndex:
            MethylationIndexList.append(c)

    print('{} of 480 cell line have gene expression data'.format(geneExpIndexList.__len__()))
    print('{} of 480 cell line have CNV data'.format(CNVIndexList.__len__()))
    print('{} of 480 cell line have miRNA data'.format(miRNAIndexList.__len__()))
    print('{} of 480 cell line have Methylation data'.format(MethylationIndexList.__len__()))

    endCellLineNameList = []
    for c in cellLineNameList['0']:
        if c in geneExpIndexList and c in CNVIndexList and c in miRNAIndexList and c in MethylationIndexList:
            endCellLineNameList.append(c)
    print('The result was {} cell lines with all four features'.format(endCellLineNameList.__len__()))
    pd.DataFrame(endCellLineNameList).to_csv('../RawData/Cell/cellLineName415.csv', header=['cellLineName'],
                                             index=True, index_label='index')


# 统计415种细胞系在反应表的480种中有多少种
def com415in480Number():
    cell415LineNameList = (pd.read_csv('../RawData/Cell/cellLineName415.csv').iloc[:, 0]).tolist()
    cell388LineNameList = (pd.read_csv('../RawData/Cell/388-cell-line-list.csv').iloc[:, 0]).tolist()
    cellLineNameList = (pd.read_csv('../RawData/Cell/cellLineNameList.csv').iloc[:, 0]).tolist()
    sumNumber = 0
    for c in cellLineNameList:
        if c in cell415LineNameList:
            print(c)
            sumNumber += 1
    print(sumNumber)


def read_cell_line_meth_feature(filename, saveFile,
                                cell_line_Name_list):  # load one of the features of cell line - copynumber
    print('{}文件开始读取------------'.format(filename))
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    features = [list() for i in range(len(cell_line_Name_list))]
    for line in reader:
        if line[0] in cell_line_Name_list:
            if line.count('NA') > 20587 * 0.2:
                print('{}号细胞系NA值超标！'.format(cell_line_Name_list.index(line[0])))
            naLocation = []
            sumMeth = 0
            for index, k in enumerate(line[1:]):
                if k == 'NA':
                    naLocation.append(index + 1)
                else:
                    sumMeth = sumMeth + float(k)
            naRe = sumMeth / (len(line) - 1 - line.count('NA'))
            for i in naLocation:
                line[i] = naRe
            features[cell_line_Name_list.index(line[0])] = line[1:]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    features = min_max_scaler.fit_transform(features)  # 最大最小归一化
    all_data = torch.FloatTensor(features)
    torch.save(all_data, saveFile)
    print('{}文件读取后被处理为{}文件，shape={}'.format(filename, saveFile, all_data.shape))


def read_cell_line_feature(filename, saveFile,
                           cell_line_Name_list):  # load one of the features of cell line - copynumber
    print('{}文件开始读取------------'.format(filename))
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    features = [list() for i in range(len(cell_line_Name_list))]
    for line in reader:
        if line[0] in cell_line_Name_list:
            features[cell_line_Name_list.index(line[0])] = line[1:]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    features = min_max_scaler.fit_transform(features)  # 最大最小归一化
    all_data = torch.FloatTensor(features)
    torch.save(all_data, saveFile)
    print('{}文件读取后被处理为{}文件，shape={}'.format(filename, saveFile, all_data.shape))


def getFourCellLineFeature():
    cellLineName415 = pd.read_csv('../RawData/Cell/cellLineName415.csv')
    geneExp = '../RawData/Cell/Gene Expression.csv'
    CNV = '../RawData/Cell/Copy Number Alterations.csv'
    miRNA = '../RawData/Cell/miRNA.csv'
    Methylation = '../RawData/Cell/Methylation.csv'
    GeneExpressionFeature402 = '../RawData/Cell/GeneExpressionFeature402.pt'
    CNVFeature402 = '../RawData/Cell/CNVFeature402.pt'
    miRNAFeature402 = '../RawData/Cell/miRNAFeature402.pt'
    MethylationFeature402 = '../RawData/Cell/MethylationFeature402.pt'

    cellLineName402List = cellLineName415['cellLineName'].tolist()
    for i in [394, 358, 355, 331, 286, 224, 214, 196, 165, 162, 138, 40, 38]:
        del cellLineName402List[i]

    read_cell_line_feature(geneExp, GeneExpressionFeature402, cellLineName402List)
    read_cell_line_feature(CNV, CNVFeature402, cellLineName402List)
    read_cell_line_feature(miRNA, miRNAFeature402, cellLineName402List)
    read_cell_line_meth_feature(Methylation, MethylationFeature402, cellLineName402List)


# 获取药物信息、获取分子指纹并保存
def processingResponseDataStep2():
    data = pd.read_csv('../RawData/drug_response_data.csv')
    drug_NameInfo = data.drop_duplicates(subset='drug_name')
    drug_NameInfo = drug_NameInfo[['drug_name', 'pubchem_cid']]
    drug_NameInfo.to_csv('../RawData/drug/drugNameInfo{}.csv'.format(len(drug_NameInfo)),
                         index=False)
    drugPubChemCidList = pd.read_csv('../RawData/drug/drugNameInfo1353.csv')['pubchem_cid'].tolist()
    fingerPrint = pd.DataFrame([])
    for index, d in enumerate(drugPubChemCidList):
        if index % 300 == 0 and index != 0:
            fingerPrint.to_csv('../RawData/drug/fingerPrintFeature{}.csv'.format(index), index=False)
        print('{}号药物开始处理---'.format(index))
        c = pcp.Compound.from_cid(d)
        fp = c.cactvs_fingerprint
        split_number = [int(char) for char in fp]
        fingerPrint = pd.concat([fingerPrint, pd.DataFrame(split_number).T])
    fp = torch.tensor(fingerPrint.to_numpy())
    torch.save(fp, '../RawData/drug/fingerPrintFeature1353.pt')


# 获取分子指纹并保存
def processingSMILESData():
    drugPubChemCidList = pd.read_csv('../data/drug/drugNameInfo1353.csv')['pubchem_cid'].tolist()
    drug_SMILES = pd.DataFrame([])
    drug_SMILES_graph = pd.DataFrame([])
    for index, d in enumerate(drugPubChemCidList):
        if index % 300 == 0 and index != 0:
            drug_SMILES.to_csv('../data/drug/SIMLESFeature{}.csv'.format(index), index=False)
        print('{}号药物开始处理---'.format(index))
        c = pcp.Compound.from_cid(d)
        sm = c.isomeric_smiles
        c_size, features, edge_index = smile_to_graph(sm)
        drug_SMILES = pd.concat([drug_SMILES, pd.DataFrame([sm]).T])
        drug_SMILES_graph = pd.concat([drug_SMILES_graph, pd.DataFrame([c_size, features, edge_index]).T])

    torch.save(drug_SMILES, '../data/drug/SMILESFeature1353.pt')
    torch.save(drug_SMILES_graph, '../data/drug/DrugSMILESGraph1353.pt')


def buildGraph():
    Methylation = torch.load('../data/cell_line/MethylationFeature402.pt')
    CNVSampleTotal = Methylation.shape[0]
    cell_sim = torch.zeros(size=(CNVSampleTotal, CNVSampleTotal))
    cell_line_edge_index = torch.IntTensor([[], []])
    for i in range(CNVSampleTotal):
        for j in range(i):
            print('开始处理第{}条甲基化数据-------'.format(i * (i + 1) / 2 + j + 1))
            temp_sim = pearsonr(Methylation[i, :], Methylation[j, :])
            cell_sim[i][j] = np.abs(temp_sim[0])
            if np.abs(temp_sim[0]) > 0.75:
                cell_line_edge_index = torch.cat((cell_line_edge_index, torch.tensor([[i], [j]])), dim=1)
    torch.save(cell_sim, '../data/cell_line/cellLineSimMeth.pt')
    torch.save(cell_line_edge_index, '../data/cell_line/cell_line_edge_index_0.75_meth.pt')

    CNV = torch.load('../data/cell_line/CNVFeature402.pt')
    CNVSampleTotal = CNV.shape[0]
    cell_sim = torch.zeros(size=(CNVSampleTotal, CNVSampleTotal))
    cell_line_edge_index = torch.IntTensor([[], []])
    for i in range(CNVSampleTotal):
        for j in range(i):
            print('开始处理第{}条拷贝数数据-------'.format(i * (i + 1) / 2 + j + 1))
            temp_sim = pearsonr(CNV[i, :], CNV[j, :])
            # cell_sim[i][j] = np.abs(temp_sim[0])
            if np.abs(temp_sim[0]) > 0.75:
                cell_line_edge_index = torch.cat((cell_line_edge_index, torch.tensor([[i], [j]])), dim=1)
    torch.save(cell_line_edge_index, '../data/cell_line/cell_line_edge_index_0.75.pt')

    fingerprint = torch.load('../data/drug/fingerPrintFeature1353.pt')
    drugSampleTotal = fingerprint.shape[0]
    drug_sim = torch.zeros(size=(drugSampleTotal, drugSampleTotal))
    drug_edge_index = torch.IntTensor([[], []])
    for i in range(drugSampleTotal):
        for j in range(i):
            print('开始处理第{}条分子指纹数据-------'.format(i * (i + 1) / 2 + j + 1))
            temp_sim = pearsonr(fingerprint[i, :], fingerprint[j, :])
            # drug_sim[i][j] = np.abs(temp_sim[0])
            if np.abs(temp_sim[0]) > 0.75:
                drug_edge_index = torch.cat((drug_edge_index, torch.tensor([[i], [j]])), dim=1)
    torch.save(drug_edge_index, '../data/drug/drug_edge_index_0.75.pt')
    # torch.save(drug_sim, '../data/drug/drugSim.pt')


def buildDrugGraphByFingerPrint():
    fp = torch.load('../data/drug/fingerPrintFeature1353.pt')
    drugSampleTotal = fp.shape[0]
    # drug_sim = torch.zeros(size=(drugSampleTotal, drugSampleTotal))
    drug_sim = torch.load('../data/drug/drugSimFingerPrint.pt')
    drug_edge_index = torch.IntTensor([[], []])
    for i in range(drugSampleTotal):
        for j in range(i):
            print('开始处理第{}条分子指纹数据-------'.format(i * (i + 1) / 2 + j + 1))
            temp_sim = pearsonr(fp[i, :], fp[j, :])
            drug_sim[i][j] = np.abs(temp_sim[0])
            if np.abs(drug_sim[i][j]) > 0.75:
                drug_edge_index = torch.cat((drug_edge_index, torch.tensor([[i], [j]])), dim=1)
    torch.save(drug_edge_index, '../data/drug/drug_edge_index_0.75_fp.pt')
    torch.save(drug_sim, '../data/drug/drugSimFingerPrint.pt')



def buildCellGraph():
    Methylation = torch.load('../data/cell_line/MethylationFeature402.pt')

    sim = pd.DataFrame(Methylation.T).corr()
    miRNA = torch.load('../data/cell_line/512dim_geneExpression_AE.pt').cpu()
    SampleTotal = Methylation.shape[0]
    cell_sim = torch.zeros(size=(SampleTotal, SampleTotal))
    # cell_sim = torch.load('../data/cell_line/cellLineSim_Meth_and_miRNA.pt')
    cell_line_edge_index = torch.IntTensor([[], []])
    for i in range(SampleTotal):
        for j in range(i):
            print('开始处理第{}条数据-------'.format(i * (i + 1) / 2 + j + 1))

            temp_sim_2 = pearsonr(Methylation[i, :], Methylation[j, :])
            #cell_sim[i][j] = np.abs(temp_sim[0]) + np.abs(temp_sim_2[0])
            cell_sim[i][j] = np.abs(temp_sim_2[0])
            if cell_sim[i][j] > 0.75:
                cell_line_edge_index = torch.cat((cell_line_edge_index, torch.tensor([[i], [j]])), dim=1)
    torch.save(cell_sim, '../data/cell_line/cellLineSim_Meth.pt')
    torch.save(cell_line_edge_index, '../data/cell_line/cell_line_edge_index_0.75_Meth.pt')


# 数据约简  选取部分数据
def dataReduction(scale=16):
    print(1 / scale)
    np.random.seed(1)
    data = pd.read_csv('../data/drug_response_data_number_log.csv')
    N = data.shape[0] * (1 / scale)
    random_index = np.random.randint(0, data.shape[0] - 1, int(N))
    reduction_data = data.iloc[random_index]
    reduction_data.to_csv('../data/drug_response_data_number_log_1_{}.csv'.format(scale), index=False)


# 计算药物和细胞系存在的边结构
def cellDrugEdge():
    scale = 50
    cell_drug = np.zeros([402, 1353])
    data = pd.read_csv('../data/drug_response_data_number_log.csv', sep=',', index_col=None)
    cell_line_drug_edge_index = torch.IntTensor([[], []])
    for i in range(data.shape[0]):
        cell_drug[data.iloc[i, 1]][data.iloc[i, 0]] = data.iloc[i, 2]
    for col in range(1353):
        print("第{}种药物开始处理-----".format(col+1))
        currentCol = cell_drug[:, col]
        IC50List = currentCol[currentCol > 0]
        IC50List = np.sort(IC50List)
        Threshold = np.percentile(IC50List, scale)  #
        newCol1 = 1 * (currentCol < Threshold)
        newCol2 = 1 * (currentCol > 0)
        newCol = newCol1 * newCol2
        cell_drug[:, col] = newCol
    for i in range(402):
        for j in range(1353):
            if cell_drug[i, j] == 1:
                cell_line_drug_edge_index = torch.cat((cell_line_drug_edge_index, torch.tensor([[i], [j]])), dim=1)
    torch.save(cell_line_drug_edge_index, '../data/cell_drug/cell_line_drug_edge_index_{}.pt'.format(scale/100))


def main():
    getDrugAndCellLineNames()



if __name__ == '__main__':
    main()
