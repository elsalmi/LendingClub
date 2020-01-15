---
title: Lending Club Project Overview
---

## <img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/iacs.png"> CS109A Introduction to Data Science

## <img style="float: right; padding-right: 10px; width: 150px" src="https://i.imgur.com/2ptDvXd.png">

**Harvard University**<br/>
**CSCI E-109A**<br/>
**Fall 2018**<br/>

**Group #47 Team Members**:<br/> 
Victor Chen<br/>
Danielle Crumley<br/>
Mohamed Elsalmi<br/>
Hoon Kang<br/>

<hr style="height:1.5pt">


## <font color='maroon'>Background Information</font>

The Lending Club is a peer-to-peer lending network for loans ranging from `$1,000` to `$40,000`. The Lending Club provides a large amount of loan data online so that its investors can make their own informed investment decisions. This online data dates from 2007 to Q2 2018 and provides certain borrower demographic and credit history information, as well as loan-specific information (including the purpose of the loan, the amount requested and amount funded, the interest rate, and the loan grade assigned by the Lending Club).

## <font color='maroon'>Project Aim</font>

The main aim of this project is to develop a model that will predict whether a Lending Club-approved loan will end up being fully paid or charged off. These predictions, of course, would be very useful to investors in determining whether or not to invest. We aim to develop a model that is both 1) accurate/effective in producing a useful prediction for investors and 2) non-discriminatory with regard to demographic features such as race and home address.

We have chosen to include only completed loans in our model (i.e., none that are still in progress, whether current or in default, since we don’t know the final outcome of these loans).


## <font color='maroon'>Data</font>


We used the aforementioned data that is available for download from the Lending Club website (https://www.lendingclub.com/).

Unfortunately, the Lending Club data for rejected loans is much more limited in scope: it includes only a few feature columns such as risk score, debt-to-income ratio, zip code, and length of employment. Since there is too much missing data, our final model will be built using only the data from the loans that were approved by the Lending Club.

First, we downloaded all of the available data on funded loans, which dates from 2007 to Q3 of 2018. It includes 145 columns and over 2 million rows.  We did not use the lending club’s data on rejected loans, since these datasets have much less information for each borrower/loan. 

Note that we were not able to incorporate borrower FICO scores into our model, since the Lending club restricts access to this information to its approved investors. However, there are many other credit risk-related features in the data, including many variables that come from the credit report such as the number of delinquencies in the borrower’s credit file in the past 2 years, the average current balance on all accounts, and the total credit revolving balance. Additionally, the lending club assigns its own loan grade (and loan subgrade) based on the FICO score and other variables, and we do have access to the Lending Club’s loan grades.

## <font color='maroon'>Methodology and Results</font>
Our data explorataion, cleaning, processing, and modeling and fairness adjustment methodologies are discussed in detail on the relevant pages of the dataset which are linked at the top of this webpage. We summarize results and future considerations on the Discussions page.

## <font color='maroon'>Sources</font>


**Kamiran, F. & Calders, T. Knowl Inf Syst (2012) 33: 1. https://doi.org/10.1007/s10115-011-0463-8**

This paper provides a method that allows us to debias a dataset exhibiting unlawful discrimination. It shows how to feed a data into a model that does not exhibit this discrimination whilst optimizing accuracy. In this paper, we concentrate on the case with only one binary sensitive attribute and a two-class classification problem we study the theoretically optimal trade-off between accuracy and non-discrimination for pure classifiers. 

**Bellamy, Rachel K.E., et al. AI Fairness 360: An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias. 3 Oct. 2018, doi:https://arxiv.org/abs/1810.01943.**

Fairness is an increasingly important concern as machine learning models are used to support decision making in high-stakes applications such as mortgage lending, hiring, and prison sentencing. This paper introduces a new open source Python toolkit for algorithmic fairness, AI Fairness 360 (AIF360), released under an Apache v2.0 license {this https URL). The main objectives of this toolkit are to help facilitate the transition of fairness research algorithms to use in an industrial setting and to provide a common framework for fairness researchers to share and evaluate algorithms. 
The package includes a comprehensive set of fairness metrics for datasets and models, explanations for these metrics, and algorithms to mitigate bias in datasets and models. It also includes an interactive Web experience (this https URL) that provides a gentle introduction to the concepts and capabilities for line-of-business users, as well as extensive documentation, usage guidance, and industry-specific tutorials to enable data scientists and practitioners to incorporate the most appropriate tool for their problem into their work products. The architecture of the package has been engineered to conform to a standard paradigm used in data science, thereby further improving usability for practitioners. Such architectural design and abstractions enable researchers and developers to extend the toolkit with their new algorithms and improvements, and to use it for performance benchmarking. A built-in testing infrastructure maintains code quality.


**Emekter, Riza, et al. Evaluating Credit Risk and Loan Performance in Online Peer-to-Peer (P2P) Lending. Applied Economics, vol. 47, no. 1, 2014, pp. 5470., doi:10.1080/00036846.2014.962222.**

This paper sheds light on P2P lending, utilizing this paper we were able to better understand how websites such as The Lending Club works and what factors would be influential in the evaluation of credit risk. 


**ONeil, Cathy. Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy. Penguin Books, 2018.**

O'Neil, a mathematician, analyses how the use of big data and algorithms in a variety of fields, including insurance, advertising, education, and policing, can lead to decisions that harm the poor, reinforce discrimination, and amplify inequality. This book has been especially helpful in informing us how modern-day credit score systems can be severely biased and discriminatory. Having a high-level of understanding of the topic assisted us in making several key decisions in designing our model and addressing fairness.  








