# Library Import #

require(devtools)
library(tm)
library(wordcloud)
library(ggplot2)
library(stm)
library(lubridate)
library(RSQLite)
library(stringr)
library(stringi)
library(ldatuning)
library(topicmodels)
library(quanteda)
library(parallel)
library(topicmodels)
library(doParallel)
library(ggplot2)
library(scales)
library(xlsx)
library(irr)
library(ggthemes)
library(lmtest)
library(tidytext)
library(tidyverse)
library(rlm)
library(lm.beta)
library(sqliter)
library(caret)
library(car)
library(wordcloud)
library(stm)
library(lubridate)
library(RSQLite)
library(lsa)
library(dplyr)
library(igraph)
library(sandwich)
library(caret)

#### Data Cleaning ####
# converting encoding
data$text<-iconv(data$text, "UTF-8", "ASCII")
dataBU2<-data
# removing short docs
removed_short<-subset(data,nchar(as.character(data$text))<500)
data2<-subset(data,!nchar(as.character(data$text))<500)

# removing congerssional documents in data
data2new<-data2[data2$source!="Congressional Documents and Publications",]

# Verifying articles mentioned associated candidate at least twice
FindMentions<-function(keys,texts) {
  length(gregexpr(strsplit(keys,"_")[[1]][length(strsplit(keys,"_")[[1]])],texts)[[1]])
}
data2new<-cbind(data2,mentions=mapply(FindMentions, data2$key, data2$text))
data2new1<-data2new[data2new$mentions>1,]

# removing duplicate entries for topic modeling (returned later)
removed_df<-data2new1[duplicated(data2new1$text),]
data3 <- data2new1[!duplicated(data2new1$text),]

#### Running Topic Models ####
# sampling 20% for k optimization #
data3_20perc_numbers<-sample((1:nrow(data3)),(nrow(data3)/5),replace=FALSE)
data3_20perc<-data3[data3_20perc_numbers,]

# creating a corpus object
mycorpus <- corpus(data3_20perc)

stopwords_and_single<-c(stopwords("english"),LETTERS,letters)
dfm_counts <- dfm(mycorpus,tolower = TRUE, remove_punct = TRUE,remove_numbers=TRUE, 
                  remove = stopwords_and_single,stem = FALSE,
                  remove_separators=TRUE) 

dfm_counts2<-dfm_trim(dfm_counts, max_docfreq = 0.95, min_docfreq=0.005,docfreq_type="prop")
dtm_lda <- convert(dfm_counts2, to = "topicmodels")

full_data<-dtm_lda

n <- nrow(full_data)

memory.limit(120000)

# optimizing k on perplxity #

print(Sys.time())
MainresultDF<-data.frame(k=c(1),perplexity=c(1),myalpha=c("x"))
MainresultDF<-MainresultDF[-1,]
candidate_alpha<- c(50) # we choose variaty of alphas
candidate_k <- c(seq(1,20)*5) # candidates for how many topics

for (eachalpha in candidate_alpha) { 
  print ("now running ALPHA:")
  print (eachalpha)
  print(Sys.time())
  #----------------5-fold cross-validation, different numbers of topics----------------
  cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
  registerDoParallel(cluster)
  
  clusterEvalQ(cluster, {
    library(topicmodels)
  })
  
  folds <- 5
  splitfolds <- sample(1:folds, n, replace = TRUE)

  clusterExport(cluster, c("full_data", "splitfolds", "folds", "candidate_k"))
  
  system.time({
    results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
      k <- candidate_k[j]
      print(k)
      results_1k <- matrix(0, nrow = folds, ncol = 2)
      colnames(results_1k) <- c("k", "perplexity")
      for(i in 1:folds){
        train_set <- full_data[splitfolds != i , ]
        valid_set <- full_data[splitfolds == i, ]
        
        fitted <- LDA(train_set, k = k, method = "Gibbs",

                      control = list(alpha=eachalpha/k) )
        results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
      }
      return(results_1k)
    }
  })
  stopCluster(cluster)
  
  results_df <- as.data.frame(results)
  results_df$myalpha<-as.character(eachalpha)
  MainresultDF<-rbind(MainresultDF,results_df)
}

print ("DONE!!!")
print(Sys.time())

# examining results
MainresultDF$kalpha=paste0(as.character(MainresultDF$k),MainresultDF$myalpha) 
ggplot(MainresultDF) +geom_boxplot(aes(x=k, y=perplexity, group=kalpha,color=myalpha))

# adding higher k values

candidate_alpha<- c(50)
candidate_k <- c(130,140,150,160,170) # candidates for how many topics

for (eachalpha in candidate_alpha) { 
  print ("now running ALPHA:")
  print (eachalpha)
  print(Sys.time())
  cluster <- makeCluster(detectCores(logical = TRUE) - 6) # leave one CPU spare...
  registerDoParallel(cluster)
  
  clusterEvalQ(cluster, {
    library(topicmodels)
  })
  
  folds <- 5
  splitfolds <- sample(1:folds, n, replace = TRUE)

  clusterExport(cluster, c("full_data", "splitfolds", "folds", "candidate_k"))
  
  system.time({
    results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
      k <- candidate_k[j]
      results_1k <- matrix(0, nrow = folds, ncol = 2)
      colnames(results_1k) <- c("k", "perplexity")
      for(i in 1:folds){
        train_set <- full_data[splitfolds != i , ]
        valid_set <- full_data[splitfolds == i, ]
        
        fitted <- LDA(train_set, k = k, method = "Gibbs",

                      control = list(alpha=eachalpha/k) )
        
        results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
      }
      return(results_1k)
    }
  })
  stopCluster(cluster)
  
  NEWresults_df <- as.data.frame(results)
  NEWresults_df$myalpha<-as.character(eachalpha)
  MainresultDF$kalpha<-paste0(as.character(MainresultDF$k),MainresultDF$myalpha)  
  NEWresults_df$kalpha<-paste0(as.character(NEWresults_df$k),NEWresults_df$myalpha) 
  MainresultDF<-rbind(MainresultDF,NEWresults_df)
}

# examinig results #

ggplot(MainresultDF) +
  geom_boxplot(aes(x=k, y=perplexity, group=kalpha,color=myalpha))+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.5)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.2)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.1)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.05)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.01)]),linetype = "dotted")

# adding higher k values

candidate_alpha<- c(50)
candidate_k <- c(180,190,200,210,220) # candidates for how many topics

for (eachalpha in candidate_alpha) { 
  print ("now running ALPHA:")
  print (eachalpha)
  print(Sys.time())
  cluster <- makeCluster(detectCores(logical = TRUE) - 6) # leave one CPU spare...
  registerDoParallel(cluster)
  
  clusterEvalQ(cluster, {
    library(topicmodels)
  })
  
  folds <- 5
  splitfolds <- sample(1:folds, n, replace = TRUE)

  clusterExport(cluster, c("full_data", "splitfolds", "folds", "candidate_k"))

  system.time({
    results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
      k <- candidate_k[j]
      results_1k <- matrix(0, nrow = folds, ncol = 2)
      colnames(results_1k) <- c("k", "perplexity")
      for(i in 1:folds){
        train_set <- full_data[splitfolds != i , ]
        valid_set <- full_data[splitfolds == i, ]
        
        fitted <- LDA(train_set, k = k, method = "Gibbs",

                      control = list(alpha=eachalpha/k) )
        
        results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
      }
      return(results_1k)
    }
  })
  stopCluster(cluster)
  
  NEWresults_df <- as.data.frame(results)
  NEWresults_df$myalpha<-as.character(eachalpha)
  MainresultDF$kalpha<-paste0(as.character(MainresultDF$k),MainresultDF$myalpha)  
  NEWresults_df$kalpha<-paste0(as.character(NEWresults_df$k),NEWresults_df$myalpha) 
  MainresultDF<-rbind(MainresultDF,NEWresults_df)
}
print(Sys.time())

# examing final results
ggplot(MainresultDF) +
  geom_boxplot(aes(x=k, y=perplexity, group=kalpha,color=myalpha))+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.5)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.2)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.1)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.05)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.01)]),linetype = "dotted")

# finding the "elbow" point of the curve $
cars.spl <- with(MainresultDF, smooth.spline(k, perplexity, df = 3))
with(cars, predict(cars.spl, x = MainresultDF$k, deriv = 2))

plot(with(cars, predict(cars.spl, x = MainresultDF$k, deriv = 2)), type = "l")
abline(v=80)

#### Running the final model over full data ####
# select full dataset
mycorpus <- corpus(data3)
stopwords_and_single<-c(stopwords("english"),LETTERS,letters)
dfm_counts <- dfm(mycorpus,tolower = TRUE, remove_punct = TRUE,remove_numbers=TRUE, 
                  remove = stopwords_and_single,stem = FALSE,
                  remove_separators=TRUE) 

dfm_counts2<-dfm_trim(dfm_counts, max_docfreq = 0.95, min_docfreq=0.005,docfreq_type="prop")
dtm_lda <- convert(dfm_counts2, to = "topicmodels")

# run model k=70
runsdf<-data.frame(myk=c(70))

mymodels<-list()

cluster <- makeCluster(detectCores(logical = TRUE) - 6) # leave one CPU spare...
registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})

#clusterExport(cluster, c("full_data", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))
clusterExport(cluster, c("dtm_lda","runsdf"))

system.time({
  mymodels <- foreach(j = 1:nrow(runsdf)) %dopar%{
    k_run <- runsdf[j,1]
    fitted <- LDA(dtm_lda, k = k_run, method = "Gibbs",
                  control = list(seed=21498) )
  }
})
stopCluster(cluster)

##### Topic Model Analysis ####

# First step - top texts and top words/frex
LDAlist=mymodels
datacolnum=5 # stating the text column

# FIRST PRINTING BETA AND THETA
#Get beta out give colnames from vocab, transpose so columns are topics and name topics from 1 to n
for (eachLDA in LDAlist)  {
  LDAfit<-eachLDA
  
  mybeta<-data.frame(LDAfit@beta)
  colnames(mybeta)<-LDAfit@terms
  mybeta<-t(mybeta)
  colnames(mybeta)<-seq(1:ncol(mybeta))
  mybeta=exp(mybeta)
  
  # Now we cycle and print top words for each topic
  nwords=50
  
  topwords <- mybeta[1:nwords,]
  for (i in 1:LDAfit@k) {
    tempframe <- mybeta[order(-mybeta[,i]),]
    tempframe <- tempframe[1:nwords,]
    tempvec<-as.vector(rownames(tempframe))
    topwords[,i]<-tempvec
  }
  
  rownames(topwords)<-c(1:nwords)
  
  kalpha<-paste0(as.character(LDAfit@k),"_",gsub("\\.","",as.character(LDAfit@alpha)))
  write.xlsx(topwords, paste0(kalpha,"_ALLCANDS_Topwords.xlsx"))
  
  # FREX words 
  # get the beta
  mybeta<-data.frame(LDAfit@beta)
  colnames(mybeta)<-LDAfit@terms
  mybeta<-t(mybeta)
  colnames(mybeta)<-seq(1:ncol(mybeta))
  mybeta=exp(mybeta)
  
  # apply formula below
  # 1/(w/(bword/sumbrow)+(1-w)/(bword)) for each cell
  myw=0.3
  word_beta_sums<-rowSums(mybeta)
  my_beta_for_frex<-mybeta
  for (m in 1:ncol(my_beta_for_frex)) {
    for (n in 1:nrow(my_beta_for_frex)) {
      my_beta_for_frex[n,m]<-1/(myw/(my_beta_for_frex[n,m]/word_beta_sums[n])+((1-myw)/my_beta_for_frex[n,m]))
    }
    print (m)
  }
  # print 50 frex:
  nwords=50
  
  topwords <- my_beta_for_frex[1:nwords,]
  for (i in 1:LDAfit@k) {
    tempframe <- my_beta_for_frex[order(-my_beta_for_frex[,i]),]
    tempframe <- tempframe[1:nwords,]
    tempvec<-as.vector(rownames(tempframe))
    topwords[,i]<-tempvec
  }
  
  rownames(topwords)<-c(1:nwords)
  
  kalpha<-paste0(as.character(LDAfit@k),"_",gsub("\\.","",as.character(LDAfit@alpha)))
  write.xlsx(topwords,paste0(kalpha,"_ALLCANDS_TopFREX.xlsx"))
  
  # TOP TEXTS --->
  data33<-data3
  data33$index<-data33$index+1
  deleted_lda_texts<-(setdiff(data33$index, as.numeric(LDAfit@documents)))
  
  '%!in%' <- function(x,y)!('%in%'(x,y))
  
  data33<-data33[data33$index %!in% deleted_lda_texts,]
  
  metadf<-data33
  meta_theta_df<-cbind(metadf[datacolnum],LDAfit@gamma)
  
  ntext=50
  
  toptexts <- mybeta[1:ntext,]
  for (i in 1:LDAfit@k) {
    print(i)
    tempframe <- meta_theta_df[order(-meta_theta_df[,i+1]),]
    tempframe <- tempframe[1:ntext,]
    tempvec<-as.vector(tempframe[,1])
    toptexts[,i]<-tempvec
  }
  
  rownames(toptexts)<-c(1:ntext)
  
  kalpha<-paste0(as.character(LDAfit@k),"_",gsub("\\.","",as.character(LDAfit@alpha)))
  write.xlsx(toptexts, paste0(kalpha,"_ALLCANDS_TopTexts.xlsx"))
  
}


#### Creating Network #### For a functionized form of ANTMN see: https://github.com/DrorWalt/ANTMN 
LDAfit<-mymodels

# remove docs deleted in topmod process due to sparcity of data
data33<-data3
data33$index<-data33$index+1
deleted_lda_texts<-(setdiff(data33$index, as.numeric(LDAfit@documents)))

'%!in%' <- function(x,y)!('%in%'(x,y))

data33<-data33[data33$index %!in% deleted_lda_texts,]

metadf<-data33
meta_theta_df<-cbind(metadf,LDAfit@gamma)

# Add size instead of person (person was a boilerplate variable)
colnames(meta_theta_df)[8]<-"size"
dfm_forsize<-data.frame(dfm_counts2)

deleted_lda_texts<-(setdiff(as.character(as.integer(dfm_forsize$document)),as.character(LDAfit@documents)))
'%!in%' <- function(x,y)!('%in%'(x,y))
dfm_forsize<-dfm_forsize[dfm_forsize$document %!in% deleted_lda_texts,]
dfm_forsize<-dfm_forsize[,-1]
sizevect<-rowSums(dfm_forsize)
meta_theta_df[,8]<-sizevect

# Add duplicate articles reviously removed for topic modeling
new_removed_df<-removed_df

colnames(new_removed_df)<-c("index.1","key.1","year.1","name.1","text.1","date.1","source.1","person.1","mentions.1")
dflist<-list()
x=0
for (i in (1:nrow(new_removed_df))) {
  x=x+1
  the_match<-match(new_removed_df$text.1[i],meta_theta_df$text)
  newvect<-c(new_removed_df[i,],meta_theta_df[the_match,])
  dflist[[x]]<-newvect
}

maintable<-data.frame(do.call(bind_rows,dflist))
maintable<-maintable[,c(c(1:7),c(17),c(9),c(19:ncol(maintable)))]
colnames(maintable)<-gsub("\\.1","",colnames(maintable))
colnames(maintable)[10:ncol(maintable)]<-seq(1:LDAfit@k)
meta_theta_df<-bind_rows(meta_theta_df,maintable)

# Add topic data for later networking
topic_data<-c(1:LDAfit@k)
topic_data<-data.frame(topic_data)
topic.frequency <- colSums(meta_theta_df[,10:ncol(meta_theta_df)]*as.vector(meta_theta_df[,8]))
topic.proportion <- topic.frequency/sum(topic.frequency)
topic_data$sumvec<-topic.proportion

library(rgexf)
# Theta/Doc network
# creating doc-topic matrix network
# now we turn to the cosine similarity calculation

mydf2<-meta_theta_df[,10:ncol(meta_theta_df)]

mycosine<-cosine(as.matrix(mydf2))
colnames(mycosine)<-colnames(mydf2)
rownames(mycosine)<-colnames(mydf2)

sem_net_weighted2<-graph.adjacency(mycosine,mode="undirected",weighted=T,diag=F,add.colnames="label") # Assign colnames
V(sem_net_weighted2)$name<-V(sem_net_weighted2)$label
V(sem_net_weighted2)$sumvec<- topic_data$sumvec
newg<-sem_net_weighted2

# printing the nets with various community detection memberships
mywalktrap<-(cluster_walktrap(newg)) 
V(newg)$walktrap<-mywalktrap$membership 

saveAsGEXF(newg, "SENATE_STRAT_THETA_cosnet.gexf")

# Preparing data for analysis and calculating share of the two communities out of news covearge for all Senate candidates

# Calculate weighted communities (not used in Walter & Ophir, 2020)
# add weights by inside edges divided by outside edges
# then do group by weight for the columns in theta
weightg<-newg
sum_degree<-sum(graph.strength(newg))
tempg1<-delete.vertices(newg,which(V(newg)$walktrap==2))
tempg2<-delete.vertices(newg,which(V(newg)$walktrap==1))
edgecalc<-data.frame(inside=c(strength(tempg1),strength(tempg2)))
edgecalc$node<-as.numeric(rownames(edgecalc))
edgecalc<-edgecalc[order(edgecalc$node),]
edgecalc$all<-strength(newg)
edgecalc$outside<-edgecalc$all-edgecalc$inside
edgecalc$community<-V(newg)$walktrap
edgecalc$commstrength<-edgecalc$inside/edgecalc$all
 
edgecalc1<-edgecalc[edgecalc$community==1,]
edgecalc2<-edgecalc[edgecalc$community==2,]

meta_theta_df1<-meta_theta_df[,which(colnames(meta_theta_df) %in% as.character(edgecalc1$node))]
meta_theta_df2<-meta_theta_df[,which(colnames(meta_theta_df) %in% as.character(edgecalc2$node))]

meta_theta_df$comm1Raw<-rowSums(meta_theta_df1)
meta_theta_df$comm2Raw<-rowSums(meta_theta_df2)

meta_theta_df1<-t(t(meta_theta_df1)*edgecalc1$commstrength)
meta_theta_df2<-t(t(meta_theta_df2)*edgecalc1$commstrength)

meta_theta_df$comm1Weighted<-rowSums(meta_theta_df1)
meta_theta_df$comm2Weighted<-rowSums(meta_theta_df2)

meta_theta_dfcomm_by_key<-aggregate(meta_theta_df[,c("comm1Raw","comm2Raw","comm1Weighted","comm2Weighted")],by=list(meta_theta_df$key),FUN=mean)
colnames(meta_theta_dfcomm_by_key)[1]<-"key"

meta_theta_df_forvol<-meta_theta_df[,1:3]
meta_theta_df_forvol$forcount<-1
meta_theta_dfvol_by_key<-aggregate(meta_theta_df_forvol[,c("forcount")],
                                   by=list(meta_theta_df_forvol$key),FUN=sum)
colnames(meta_theta_dfvol_by_key)[1]<-"key"
colnames(meta_theta_dfvol_by_key)[2]<-"volume"

finaldf<-cbind(meta_theta_dfcomm_by_key,meta_theta_dfvol_by_key)
newdata<-finaldf
write.csv(finaldf,"meta_theta_dfcomm_by_key.csv")


#### MAIN DATA ANALYSIS FOR WALTER & OPHIR, 2020 ####
# cleaning and organizing the data for regression analysis
newdata<-read.csv("meta_theta_dfcomm_by_key.csv")   
repdata<-newdata[newdata$PARTY=="R",]
demdata<-newdata[newdata$PARTY=="D",]
repdata<-repdata[order(repdata$dyad),]
demdata<-demdata[order(demdata$dyad),]

n<-data.frame(repdata,demdata)

attach(n)

n$EXPERIENCE <- relevel(as.factor(n$EXPERIENCE),ref = 3)
n$EXPERIENCE.1 <- relevel(as.factor(n$EXPERIENCE.1),ref = 3)

attach(n)

colnames(n)<-c("r.KEY","r.inc","r.year","r.STATE","r.CANDIDATE","r.gender","r.fund",
               "r.exp","r.exp_yrs","r.conserv","r.midterm","r.party",
               "r.percvotes","r.strat","r.issue","r.strat_weighted","r.issue_wighted","r.volume","r.dyad","r.yearborn",
               "d.KEY","d.inc","d.year","d.STATE","d.CANDIDATE","d.gender","d.fund",
               "d.exp","d.exp_yrs","d.conserv","d.midterm","d.party",
               "d.percvotes","d.strat","d.issue","d.strat_weighted","d.issue_wighted","d.volume","d.dyad","d.yearborn")
attach(n)

# Experience as dummies
dummies <- data.frame(predict(dummyVars(~ r.exp, data = n), newdata = n))
dummies2 <- data.frame(predict(dummyVars(~ d.exp, data = n), newdata = n))

n<-data.frame(n,dummies,dummies2)

rownames(n) <- NULL

n$r.percvotes<-n$r.percvotes/100

attach(n)

#### REGRSSION MODELS #### 

race_data <- n
attach(race_data)

## Minimal parsimonous model (only framing and votes)
ggplot(newdata,aes(x=comm1Raw,y=GENERAL_PERC))+
  geom_point(aes(color=PARTY))+
  scale_color_manual(values=c("skyblue3", "lightcoral"))+
  ylab("Percentage of Votes Gathered By Candidate")+
  xlab("Stratergy Share of News Coverage of Candidate")+
  geom_smooth(method="lm",aes(y=GENERAL_PERC,color=PARTY),alpha=0.15)+
  theme_bw()

minimal_model<-lm(r.percvotes~r.strat+d.strat,data=race_data)
vif(minimal_model)
bptest(minimal_model)
summary((minimal_model))
summary(lm.beta(minimal_model))
model1<-minimal_model

vcv <- vcovHC(model1, type = "HC1")
coeftest(model1, vcv)


## model with only the controls
model_control<-lm((r.percvotes)~(r.midterm)+(r.conserv)+r.gender+
                    (r.exp.senator)+(r.exp.congressman)+(r.exp.governor)+(r.exp.other)+
                    (r.fund)+I(r.fund^2)+r.volume
                  +d.gender+
                    (d.exp.senator)+(d.exp.congressman)+(d.exp.governor)+(d.exp.other)+
                    (d.fund)+I(d.fund^2)+d.volume
                  ,data=race_data)
vif(model_control)
bptest(model_control)
summary(lm.beta(model_control))
model2<-model_control

vcv <- vcovHC(model_control, type = "HC1")
coeftest(model_control, vcv)


## Full Model
model_all<-lm((r.percvotes)~(r.midterm)+(r.conserv)+r.gender+
                (r.exp.senator)+(r.exp.congressman)+(r.exp.governor)+(r.exp.other)+
                (r.fund)+I(r.fund^2)+r.volume
              +d.gender+
                (d.exp.senator)+(d.exp.congressman)+(d.exp.governor)+(d.exp.other)+
                (d.fund)+I(d.fund^2)+d.volume+
                (r.strat)+(scale(d.strat))
              ,data=race_data)

vif(model_all)
bptest(model_all)
summary(lm.beta(model_all))
model3<-model_all

library(sandwich)
vcv <- vcovHC(model_all, type = "HC1")
coeftest(model_all, vcv)

# print results
stargazer(model2,model1,model3, type="html",out="filetopaste.html")



#### creating APPENDIX 3 #### same analysis but without candidates' subsequent runs 
# delete couble cands
no_dup_n<-race_data[!duplicated(race_data$r.CANDIDATE),]
no_dup_n<-no_dup_n[!duplicated(no_dup_n$d.CANDIDATE),]

par(mfrow=c(1,2))

# running original
p1<-ggplot(newdata,aes(x=comm1Raw,y=GENERAL_PERC))+
  geom_point(aes(color=PARTY))+
  scale_color_manual(values=c("skyblue3", "lightcoral"))+
  ylab("Percentage of Votes Gathered By Candidate")+
  xlab("Stratergy Share of News Coverage of Candidate")+
  geom_smooth(method="lm",aes(y=GENERAL_PERC,color=PARTY),alpha=0.15)+
  theme_bw()

p2<-ggplot(no_dup_newdata,aes(x=comm1Raw,y=GENERAL_PERC))+
  geom_point(aes(color=PARTY))+
  scale_color_manual(values=c("skyblue3", "lightcoral"))+
  ylab("Percentage of Votes Gathered By Candidate")+
  xlab("Stratergy Share of News Coverage of Candidate")+
  geom_smooth(method="lm",aes(y=GENERAL_PERC,color=PARTY),alpha=0.15)+
  theme_bw()

library(ggpubr)
ggarrange(p1,p2,
          labels = c("Original Data", "Candidates Only Appear Once"),
          ncol = 2, nrow = 1)

# Running Regression models as earlier
# minimal model
minimal_model<-lm(r.percvotes~r.strat+d.strat,data=no_dup_n)
vif(minimal_model)
bptest(minimal_model)
summary((minimal_model))
summary(lm.beta(minimal_model))
model1<-minimal_model

library(sandwich)
vcv <- vcovHC(minimal_model, type = "HC1")
coeftest(minimal_model, vcv)

# controls only model
model_control<-lm((r.percvotes)~(r.midterm)+(r.conserv)+r.gender+
                    (r.exp.senator)+(r.exp.congressman)+(r.exp.governor)+(r.exp.other)+
                    (r.fund)+I(r.fund^2)+r.volume
                  +d.gender+
                    (d.exp.senator)+(d.exp.congressman)+(d.exp.governor)+(d.exp.other)+
                    (d.fund)+I(d.fund^2)+d.volume
                  ,data=no_dup_n)
vif(model_control)
bptest(model_control)
summary(lm.beta(model_control))

model2<-model_control

library(sandwich)
vcv <- vcovHC(model_control, type = "HC1")
coeftest(model_control, vcv)

# full model
model_all<-lm((r.percvotes)~(r.midterm)+(r.conserv)+r.gender+
                (r.exp.senator)+(r.exp.congressman)+(r.exp.governor)+(r.exp.other)+
                (r.fund)+I(r.fund^2)+r.volume
              +d.gender+
                (d.exp.senator)+(d.exp.congressman)+(d.exp.governor)+(d.exp.other)+
                (d.fund)+I(d.fund^2)+d.volume+
                (r.strat)+((d.strat))
              ,data=no_dup_n)

vif(model_all)
bptest(model_all)
summary(lm.beta(model_all))

model3<-model_all

library(sandwich)
vcv <- vcovHC(model_all, type = "HC1")
coeftest(model_all, vcv)


ggplot(n,aes(y=r.strat,x=d.strat))+geom_point()+
  ylab("Republican Share of Strategy Oriented News COverage")+
  xlab("Democrat Share of Strategy Oriented News COverage")+
  theme_bw()
cor.test(n$r.strat,n$d.strat)


#### PRINTING ALL NEWS SOURCE - APPENDIX 5 ####

sources_to_print<-data.frame(table(meta_theta_df$source))
colnames(sources_to_print)<-c("Outlet","# of Articles")
write.xlsx(sources_to_print,"sources_table.xlsx")
