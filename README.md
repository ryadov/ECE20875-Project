# Final Project
## Due: 4/22/2022 11:59pm ET

Up until now, we have given you fairly detailed instructions for how to design data analyses to answer specific questions about data -- in particular, how to set up a particular analysis and what steps to take to run it. In this project, you will put that knowledge to use!

Put yourself in the shoes of a data scientist that is given a data set and is asked to draw conclusions from it. Your job will be to understand what the data is showing you, design the analyses you need, justify those choices, draw conclusions from running the analyses, and explain why they do (or do not) make sense.

We are deliberately not giving you detailed directions on how to solve these problems, but feel free to come to office hours to brainstorm.

## Objectives

There are two possible paths through this project:

1. You may use data set #1, which captures information about bike usage in New York City. See below for the analysis questions we want you to answer.
2. You may use data set #2, which captures information about student behavior and performance in an online course. See below for the analysis questions we want you to answer.

**Note that you can select EITHER data set #1 or data set #2, but you should NOT complete the mini-project for both dataset.** 

We have provided starter codes to help you load the data to your environment using `MiniProjectPath1.py` and `MiniProjectPath2.py` for paths 1 and 2 respectively. The code provided is supplemental material to help you with with the data loading process, however feel free to choose a different method to read the data. It is not mandatory to use this code.  

In your final report, you are expected to answer the questions asked below corresponding to your chosen dataset. For both data sets, please begin by including a section on descriptive statistics (see details below). In addition, please justify why you chose to use certain methods and models to answer the questions (for the most part, there will not be incorrect answers regarding a particular model as long as your reasoning is explained).

## Partners

On this project **you may work with one partner** (except for Honors contracting students who must work individually). Working with a partner is optional, and working with a partner will not impact how the project is graded. If you want to work with a partner, it is your responsibility to pair up; feel free to use Piazza's "Search for Teammates" feature (https://piazza.com/class/ky202w8tbhxbw?cid=5) to facilitate this. **You may only pair with students who are in the SAME SECTION as you**.

If you are working with a partner, _you must coordinate with your partner_. This means that each one of you _can_ clone your repository on GitHub classroom, but only one repository will be graded (the leader's repository). It is the responsibility of the group to look for ways to collaborate for this project. 

If you are working as a team and you are the team leader, you must fill out this form: https://forms.gle/TcKScvY8xCk9gjjs7. 

No need to fill out the form if you are working individually. 

## Descriptive statistics

For the descriptive statistics task, you need to summarize your dataset. This description will help you be more knowledgeable of your dataset and intentional in choosing your methods of data analysis when answering the questions. 
The description should include, but is not limited to (i.e., feel free to include more descriptive statistics than what is asked below, but not less): 

1. What are the variables in your dataset? What do they mean (describe the variables that you plan to use)?
2. After reading the questions for the data set you have chosen to work with, provide a summary statistics table of the variables you will use. If you need to transform a variable (e.g., Precipitation into a Raining or not raining variable), this variable must be included in the table. You can use any appropriate summary statistics (e.g., mean, standard deviation, mode).
3. Provide a histogram and explain the resulting plot for at least one variable in your dataset

_Descriptive statistics_ should be included in both paths.

## Path 1: Bike traffic

The `NYC_Bicycle_Counts_2016_Corrected.csv` gives information on bike traffic across a number of bridges in New York City. In this path, the analysis questions we would like you to answer are as follows:

1. You want to install sensors on the bridges to estimate overall traffic across all the bridges. But you only have enough budget to install sensors on three of the four bridges. Which bridges should you install the sensors on to get the best prediction of overall traffic?
2. The city administration is cracking down on helmet laws, and wants to deploy police officers on days with high traffic to hand out citations. Can they use the next day's weather forecast (low/high temperature and precipitation) to predict the total number of bicyclists that day? 
3. Can you use this data to predict whether it is raining based on the number of bicyclists on the bridges (_hint_: The variable `raining` or `not raining` is binary)?
   
## Path 2: Student performance related to video-watching behavior

`behavior-performance.txt` contains data for an online course on how students watched videos (e.g., how much time they spent watching, how often they paused the video, etc.) and how they performed on in-video quizzes. `readme.pdf` details the information contained in the data fields. There might be some extra data fields present than the ones mentioned here. Feel free to ignore/include them in your analysis. In this path, the analysis questions we would like you to answer are as follows:

(For Q2,Q3: _You will run prediction algorithm(s) for __ALL__ students for __ONE__ video, and repeat this process for all videos._ The function `get_by_VidID` in the helper file `MiniProjectPath2` will help you in this process.)

1. How well can the students be naturally grouped or clustered by their video-watching behavior (`fracSpent`, `fracComp`, `fracPaused`, `numPauses`, `avgPBR`, `numRWs`, and `numFFs`)? You should use all students that complete at least five of the videos in your analysis. Hints: Would KMeans or Gaussian Mixture Models be more appropriate? Consider using both and comparing.
2. Can student's video-watching behavior be used to predict a student's performance (i.e., average score `s` across all quizzes)?(_hint_: Just choose 1 - 4 data fields to create your model. We are looking at your approach rather than model performance.)
3. Taking this a step further, how well can you predict a student's performance on a *particular* in-video quiz question (i.e., whether they will be correct or incorrect) based on their video-watching behaviors while watching the corresponding video? You should use all student-video pairs in your analysis.

## What to turn in
You must turn in two sets of files by pushing them to your team's Github repository:

* `report.pdf`: A project report, which should consist of:
  * A section with the names of the team members (maximum of two), your Purdue username(s), and the path (1 or 2) you have taken. Use the heading "Project team information".
  * A section stating and describing the dataset you are working with. Use the heading "Descriptive Statistics".
  * A section describing the methods of data analysis you chose to use for each analysis question (with a paragraph or two justifying why you chose that method and what you expect the analysis to tell you). Use the heading "Approach".
  * A section describing the results of each method/analysis, and what your answers to the questions are based on your results. **At list one visual aid should be included** (you can use many if necessary to back up your conclusions). Nonetheless, any visual included should be mentioned in the text.  Note that it is OK if you do not get a "close" answer/answers from your analysis, but you must explain why that might be. Here, you should essentially be answering the questions asked above for your chosen data set. Use the heading "Analysis".

* All Python `.py` code files you wrote to complete the analysis steps. In addition, the report must be a PDF file. See the template provided to guide yourself on the format of the project. Not complying with the instructions might result in a deduction of points. 
