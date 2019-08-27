# Detecting_interesting_VC_investments
1.	AIM OF THE PROJECT

Given a large pool of applications and their corresponding change in ranking for the past 3 years, detect applications that are potential candidates for venture capital investments.
The project was performed on a dataset of 68,000 of real applications on the apple application store. 

![Applications by category](/images/appcat.png)

To take the project a step further the scope was also increased to check whether the growth in ranking of an application can be predicted using various available features and applying the most appropriate machine learning models available.
Further research was also conducted into various potential data sources that would increase the predictability of growth of a firm.

An acceptable working machine learning model was also created to predict consistency in performance.

2.	ANALYSIS OF THE DATA

Two raw datasets were available which contained information pertaining to applications. 

App info data: 
This first dataset contained various features which provided basic information about a particular application. This information is readily visible on the iOS app store pages. 

![App Info set](/images/app info.png)


The features are as follows:
*	itunes_app_id: The unique identification number of an application
*	app_name: The name of the application
*	developer: The developer of the application
*	website:  The website of the company of the application 
*	category: The category to which an application belongs to
*	is_editor_choice:  Whether or not an application is an editors choice app
*	rating_oo5: The user rating out of 5 achieved by application
*	num_ratings: The number of user ratings provided for the application
*	has_iap: Whether the application has in-app purchases or not
*	release_date: The date the application was released
*	current_version: The latest version of the application
*	age_rating: The age group that the app is available to
*	file_size: The file size of the application
*	editor_notes: Notes from the editor
*	description:  The description text of an application
*	os_compatibility: Operating system which the application is compatible with
*	languages: The various languages supported by the application
*	price: The cost of the application
*	itunes_link: Link to the apps iTunes page
*	date: Date of data retrieval
 
Ranking info data: 
The next dataset was more crucial and contained the ranking information of each application for the last three years. The greatest and most visible hurdle in the dataset was the presence of numerous null values. This was a significant initial barrier for the project.
However using advanced imputation methodologies, this barrier could be overcome. To get an idea of the dataset a screenshot of the ranking panel can be seen below:
 



3.	METHODOLOGY

This section discusses all of the steps followed to achieve the objective of the project.


i.	Ranking dataset Imputation: 

As it was evident during the analysis of the data, the ranking panel data frame needed to be setup properly. The null values needed to be taken care of.
Intuitively, this was achieved by first performing a forward fill to imply a conservative approach where the past ranks were carried forward. This handled consequent null values. The second step in the imputation process was to perform a “backward fill” to handle ranking panel data that started with null values. These were not filled during the forward fill process.
In this manner all null values were imputed for further analysis.


ii.	Ranking Feature Extraction:

The next step was to extract features from the ranking dataset. 
Five important app metrics were derived from the data, they are listed below:
*	Top rank achieved by the app: In the past 3 years, the highest rank achieved by an app was derived.
*	Rank growth achieved by the app: The growth of an app in terms of rank was also calculated. The formula used was:
*	Rank growth = (Worst Rank – Rank on last day)
*	Days at top rank: This feature was intended to check the consistency of an app performing well. It denotes the total number of days an application stayed at its top rank.
*	Average daily growth achieved by the app: Sometimes an app rapidly reaches its top rank and stops growing in terms of rank but in terms of revenue and company performance, the application continues to grow. This is known as the ceiling effect. To overcome this and to estimate how much an app grows, the average daily growth was computed by first deriving the age of the app from the application info data frame and then dividing rank growth by this value.
*	Maximum rate of rank growth of the app: This is the percentage of rank growth achieved by the app. This feature also intends to overcome the ceiling effect.

iii.	Other features :

*	Subscription app: Subscription services were highly favored by venture capitalists. To support this a natural language processing was done on the description column of applications that had a higher user rating than 4. The top word occurring in this analysis was the word subscription. This was a strong indication that the feature whether an app is a subscription app or not is important for potential investors. Additionally all applications that the investors indicated as being interesting for them were subscription apps. 
*	Days the App has been active: As an investor it is always a priority to invest in firms which are early in their stages and also indicate high potential for growth. The age of an application therefore becomes very interesting.
*	Category of the app: Another interesting attribute of an app with regards to investments would be whether the app is in a category where VC investments are common for the last few quarters. This indicates the current trends in the market. This was gauged from a website called CB insights.
*	Developer portfolio: The developer of an app is also really an interesting aspect. It was interesting to find out whether some developers were consistently delivering highly growing apps.



iv.	The scoring model:

All the derived categories were then combined and assigned weights. The apps were all compared using this scoring model and scores were generated across categories. The potential applications ideal for VC investments were thus found.


4. Machine learning model to predict consistency

The project was taken a step further to create a machine learning model to predict the number of days an app consistently stayed at the top rank. Machine learning models namely a Multiple linear regression model, K nearest neighbor regression, Decsion tree regression, boosted trees and finally random forest decision trees. The random forest model performed the best achieving an accuracy of close to 95%
