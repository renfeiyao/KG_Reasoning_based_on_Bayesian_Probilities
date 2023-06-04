#utils
#导入工具包，定义全局变量
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import time
from pgmpy.estimators import K2Score
from pgmpy.estimators import ExhaustiveSearch
import matplotlib.pyplot as plt
PATH='C:\\Users\\paper\\knowledge_center\\'
'''
file_name=['decision','engineering','engineering_entityrelation','engineering_historysystem','entity',
           'entity_property','entity_relation','entity_standard','entity_type','evaluation','forecast',
          'history','literature','property_standard','property_type','relation_type']
'''

def build_D():
    '''
    从entity, entity_type表中构造案例数据集D，从relation_type中提取网络结构
    '''
    #decision=pd.DataFrame(pd.read_csv(PATH+'decision'+'.csv'))
    #engineering=pd.DataFrame(pd.read_csv(PATH+'engineering'+'.csv'))
    #engineering_entityrelation=pd.DataFrame(pd.read_csv(PATH+'engineering_entityrelation'+'.csv'))
    #engineering_historysystem=pd.DataFrame(pd.read_csv(PATH+'engineering_historysystem'+'.csv'))
    entity=pd.DataFrame(pd.read_csv(PATH+'entity'+'.csv'))
    #entity_property=pd.DataFrame(pd.read_csv(PATH+'entity_property'+'.csv'))
    entity_relation=pd.DataFrame(pd.read_csv(PATH+'entity_relation'+'.csv'))
    entity_standard=pd.DataFrame(pd.read_csv(PATH+'entity_standard'+'.csv'))
    entity_type=pd.DataFrame(pd.read_csv(PATH+'entity_type'+'.csv'))
    #evaluation=pd.DataFrame(pd.read_csv(PATH+'evaluation'+'.csv'))
    #forecast=pd.DataFrame(pd.read_csv(PATH+'forecast'+'.csv'))
    #history=pd.DataFrame(pd.read_csv(PATH+'history'+'.csv'))
    #literature=pd.DataFrame(pd.read_csv(PATH+'literature'+'.csv'))
    #property_standard=pd.DataFrame(pd.read_csv(PATH+'entity_standard'+'.csv'))
    #property_type=pd.DataFrame(pd.read_csv(PATH+'entity_type'+'.csv'))
    relation_type=pd.DataFrame(pd.read_csv(PATH+'relation_type'+'.csv'))
    
    #为各实体匹配类别
    entity_to_type=pd.merge(entity,entity_standard,how='left',on='Entity_StandardID')
    entity_to_type.save(PATH+'entity_to_type.csv')
    
    #提取网络结构
    BNnodes=[]
    for i in range(len(relation_type)):
        head_type=relation_type.loc[i,'Head_EntityID']
        tail_type=relation_type.loc[i,'Tail_EntityID']
        head_type_name=entity_type[entity_type['Entity_TypeID']==head_type].iloc[0,1]
        tail_type_name=entity_type[entity_type['Entity_TypeID']==tail_type].iloc[0,1]
        if head_type=='En0015' and tail_type!='En0001':
            if tail_type!='En0014'and tail_type!='En0002':
                nodeset=(tail_type_name,head_type_name)
            else:
                nodeset=(head_type_name,tail_type_name)
        BNnodes.append(nodeset)
    
    #提取数据集
    #构造空Dataframe,列名为实体类别
    D = pd.DataFrame(columns=entity_type.Entity_TypeName[1:].tolist())
    #抽取符合类别的实体
    for i in range(len(entity_relation)):
    head_entity=entity_relation.loc[i,'Head_EntityID']
    tail_entity=entity_relation.loc[i,'Tail_EntityID']
    head_entity_type=entity_to_type[entity_to_type['Entity_ID']==head_entity].iloc[0,4]
    tail_entity_type=entity_to_type[entity_to_type['Entity_ID']==tail_entity].iloc[0,4]
    tail_type_name=entity_type[entity_type['Entity_TypeID']==tail_entity_type].iloc[0,1]
    #填入Dataframe
    if head_entity_type=='En0015':
        if head_entity not in D['Project'].values:
            D.loc[len(D)+1,'Project']=head_entity
        if tail_entity_type!='En0015'and tail_entity_type!='En0001':
            if D.loc[D['Project']==head_entity,tail_type_name].isnull().any():
                D.loc[D['Project']==head_entity,tail_type_name]=D.loc[D['Project']==head_entity,tail_type_name].fillna(value=tail_entity)
            else:
                index=len(D)+1
                D.loc[index]=D[D['Project']==head_entity].iloc[0,:]
                D.loc[index,tail_type_name]=tail_entity
    #将Nan值替换为空节点NULL
    D=D.fillna(value='NULL')
    D.to_pickle(PATH+'D.pkl')
    return BNnodes

def show_D_distribution(D):
    '''
    观察现象和措施的分布
    '''
    plt.rcParams['figure.figsize']=(10.0,5.0)
    plt.rcParams['font.sans-serif']=['KaiTi']
    plt.title('现象节点变量分布')
    plt.hist(D['Phenomenon'],bins=state_names['Phenomenon'])
    plt.savefig(PATH+'Phenomenon_hist.png',dpi=300,bbox_inches='tight')
    plt.show()
    
    plt.rcParams['figure.figsize']=(30.0,15.0)
    plt.title('措施节点变量分布')
    plt.hist(D['Target'],bins=state_names['Target'])
    plt.savefig(PATH+'Target_hist.png',dpi=300,bbox_inches='tight')
    plt.show()

#分批训练，一起训练容易卡死
def train_BN(model,D_train,batch_size):
    lines=len(D_train)-batch_size
    batch=int(lines//batch_size)
    for i in range(batch):
        model.fit_update(D_train[(i+1)*batch_size+1:(i+2)*batch_size])
        model.save(PATH+'BN_test'+str(i+2)+'.bif',filetype='bif')
        print("batch",i+1)
    model.fit_update(D_train[batch*batch_size+1:])
    print("batch",batch+1)
    return model

def get_BN():
    '''
    利用数据集和网络结构构建并训练BN
    '''
    #获取结构
    BNnodes=build_D()
    #训练数据集
    D=pd.read_pickle(PATH+'D.pkl')
    D_train, D_test=train_test_split(D, test_size=0.33, random_state=42)
    D_train.to_pickle(PATH+'D_train.pkl')
    D_test.to_pickle(PATH+'D_test.pkl')
    #状态变量集
    state_names=dict()
    for i in D.columns.tolist():
        state_names[i]=D[i].unique().tolist()
    #数据查看
    for i in state_names.keys():
        print(i,len(state_names[i]))
    #构建
    start=time.time()
    model = BayesianNetwork(BNnodes)
    batch_size=3
    #训练
    model.fit(D_train[0:batch_size], estimator=MaximumLikelihoodEstimator,state_names=state_names)
    model=train_BN(model,D_train,batch_size)
    end=time.time()
    model.save(PATH+'BN_full.bif',filetype='bif')
    print('time:',end-start)
    return model
    
def predict_nodes():
    '''
    贝叶斯网络节点变量预测
    '''
    model= BayesianNetwork.load(PATH+'BN_full.bif', filetype='bif')
    D_test=pd.read_pickle(PATH+'D_test.pkl')
    start=time.time()
    #df=pd.DataFrame(index=D_test.index)
    for i in D.columns.values:
        possibilities=model.predict_probability(D_test.drop([i],axis=1))
        possibilities.to_csv(PATH+'predict_results\\'+i+'_possibilities.csv')
        results=model.predict(D_test.drop([i],axis=1))
        results.to_csv(PATH+'predict_results\\'+i+'_results.csv')
        df.loc[:,i]=results
        print(i)
        #for t in df.index:
            #df.loc[t,i+'_possibilities']=possibilities.loc[t,i+'_'+results.loc[t,i]]
    end=time.time()
    print('time:',end-start)
    
def predict_links(types,numlimit,threshold):
    '''
    预测三元组
    '''
    df=pd.DataFrame(columns=['HeadEntity','TailEntity','Link_test','Link_pred'])
    entity_relation=pd.DataFrame(pd.read_csv(PATH+'entity_relation'+'.csv'))
    for i in types:
        possibilities=pd.DataFrame(pd.read_csv(PATH+'predict_results\\'+i+'_possibilities.csv'))
        if numlimit[i]==1:
            results=pd.DataFrame(pd.read_csv(PATH+'predict_results\\'+i+'_results.csv'))
            for t in possibilities.index:
                if possibilities.loc[t,i+'_'+results.loc[t,i]]>threshold[i]:
                    df.loc[t,i+'_possibilities']=results.loc[t,i+'_'+results.loc[t,i]]
        else:
            for t in possibilities.index:
                if possibilities.loc[t,i+'_'+results.loc[t,i]]>threshold:
                    for n in numlimit[i]:
                        df.loc[t,i+'_possibilities']=possibilities.loc[t,i+'_'+results.loc[t,i]].tolist()
                        
def showBN(model,save=True):
    '''传入BayesianModel对象，调用graphviz绘制结构图，jupyter中可直接显示'''
    from graphviz import Digraph
    node_attr = dict(
     style='filled',
     shape='box',
     align='left',
     fontsize='12',
     ranksep='0.1',
     height='0.2'
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    edges=model.edges()
    for a,b in edges:
        dot.edge(a,b)
    if save:
        #dot.view(cleanup=True)
        dot.render(directory=PATH+'doctest-output', view=True)  
    return dot
 
#没用
def structure_learning(model,D):
    '''
    贝叶斯网络结构学习
    '''
    es = ExhaustiveSearch(D, scoring_method=K2Score(D))
    best_model = es.estimate()
    print(best_model.edges())

    print("\nAll DAGs by score:")
    for score, dag in reversed(es.all_scores()):
        print(score, dag.edges())
    best_model.save(PATH+'best_model.bif',filetype='bif') 

def print_cpds(model):
    '''
    打印cpd表
    '''
    for cpd in model.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)

model=get_BN()
print_cpds(model)
D=pd.read_pickle(PATH+'D.pkl')
show_D_distribution(D)
types=pd.read_csv(PATH+'relation_types.csv')
numlimit=types.numlimit
threshold=types.threshold
predict_links(types,numlimit,threshold)
dot=showBN(model)   