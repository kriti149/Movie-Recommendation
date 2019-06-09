# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:12:56 2018

@author: Kriti Gupta
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation
path='E:/movie recommendation/ml-100k/u.data'
data=pd.read_csv(path,sep="\t",header=None,names=['userId','itemId','rating','timestamp'])
movieinfopath='E:/movie recommendation/ml-100k/u.item'
movieinfo=pd.read_csv(movieinfopath,sep="|",header=None,index_col=False,names=['itemId','title'],usecols=[0,1],encoding='latin-1')
data=pd.merge(data,movieinfo,left_on='itemId',right_on='itemId')

data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])
numofusers=max(data.userId)
numofmovies=max(data.itemId)
movieperuser=data.userId.value_counts()
userpermovie=data.title.value_counts()
def favmovies(activeuser,n):
    topmovies=pd.DataFrame.sort_values(data[data.userId==activeuser],['rating'],ascending=[0])[:n]
    return list(topmovies.title)
print(favmovies(5,3))
useritemratingmatrix=pd.pivot_table(data,values='rating',index=['userId'],columns=['itemId'])

def similarity(user1,user2):
    user1=np.array(user1)-np.nanmean(user1)
    user2=np.array(user2)-np.nanmean(user2)
    commonitemids=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
    if len(commonitemids)==0:
        return 0
    else:
        user1=np.array([user1[i] for i in commonitemids])
        user2=np.array([user2[i] for i in commonitemids])
        return correlation(user1,user2)

def nearestneighborrating(activeuser,k):
    similaritymatrix=pd.DataFrame(index=useritemratingmatrix.index,columns=['similarity'])
    for i in useritemratingmatrix.index:
        similaritymatrix.loc[i]=similarity(useritemratingmatrix.loc[activeuser],useritemratingmatrix.loc[i])
    similaritymatrix=pd.DataFrame.sort_values(similaritymatrix,['similarity'],ascending=[0])
    nearestneighbors=similaritymatrix[:k]
    neighboritemratings=useritemratingmatrix.loc[nearestneighbors.index]
    predictitemrating=pd.DataFrame(index=useritemratingmatrix.columns,columns=['rating'])
    for i in useritemratingmatrix.columns:
        predrating=np.nanmean(useritemratingmatrix.loc[activeuser])
        for j in neighboritemratings.index:
            if useritemratingmatrix.loc[j,i]>0:
                predrating+=(useritemratingmatrix.loc[j,i]-np.nanmean(useritemratingmatrix.loc[j]))*nearestneighbors.loc[j,'similarity']
        predictitemrating.loc[i,'rating']=predrating
    return predictitemrating
def topNrecommendation(activeuser,n):
    predicteditemrating=nearestneighborrating(activeuser,10)
    moviesalreadywatched=list(useritemratingmatrix.loc[activeuser].loc[useritemratingmatrix.loc[activeuser]>0].index)
    predicteditemrating=predicteditemrating.drop(moviesalreadywatched)
    toprecommendations=pd.DataFrame.sort_values(predicteditemrating,['rating'],ascending=[0])[:n]
    toprecommendationtitles=(movieinfo.loc[movieinfo.itemId.isin(toprecommendations.index)])
    return list(toprecommendationtitles.title)
activeuser=5
print(favmovies(activeuser,5))
print(topNrecommendation(activeuser,3))

#latent factor collaborative filtering
def matrixfactorization(r,k,steps=10,gamma=0.001,lamda=0.02):
    n=len(r.index)
    m=len(r.columns)
    p=pd.DataFrame(np.random.rand(n,k),index=r.index)
    q=pd.DataFrame(np.random.rand(m,k),index=r.columns)
    for step in range(steps):
        for i in r.index:
            for j in r.columns:
                if r.loc[i,j]>0:
                    eij=r.loc[i,j]-np.dot(p.loc[i],q.loc[j])
                    p.loc[i]=p.loc[i]+gamma*(eij*q.loc[j]-lamda*p.loc[i])
                    q.loc[j]=q.loc[j]+gamma*(eij*p.loc[i]-lamda*q.loc[j])
        e=0
        for i in r.index:
            for j in r.columns:
                e=e+pow(r.loc[i,j]-np.dot(p.loc[i],q.loc[j]),2)+lamda*(pow(np.linalg.norm(p.loc[i]),2)+pow(np.linalg.norm(q.loc[j]),2))
        if e<0.001:
            break
        print (step)
    return p,q
(p,q)=matrixfactorization(useritemratingmatrix,k=2,steps=100,gamma=0.001,lamda=0.02)
activeuser=5
predicteditemrating=pd.DataFrame(np.dot(p.loc[activeuser],q.T),index=q.index,columns=['rating'])
moviesalreadywatched=list(useritemratingmatrix.loc[activeuser].loc[useritemratingmatrix.loc[activeuser]>0].index)
predicteditemrating=predicteditemrating.drop(moviesalreadywatched)
toprecommendations=pd.DataFrame.sort_values(predicteditemrating,['rating'],ascending=[0])[:3]
toprecommendationtitles=(movieinfo.loc[movieinfo.itemId.isin(toprecommendations.index)])
print (list(toprecommendationtitles.title))


                    

    
    
                
                
    