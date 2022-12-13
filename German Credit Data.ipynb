{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Week 5 Exercise: German Credit Data \n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Data Preprecessing\n",
    "\n",
    "#### Q1. Given the background,  why would you create a decision tree model on this dataset?\n",
    "\n",
    "In this step, you need to read the data, have an overview of the data, and perform label encoding for both the predictors and target variable.\n",
    "\n",
    "1. first let's read the data.\n",
    "\n",
    "##### code for reference:\n",
    "import pandas as pd\n",
    "\n",
    "credit = pd.read_csv(\"your path /credit.csv\")\n",
    "\n",
    "credit.head()\n",
    "\n",
    "credit.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 35
    },
    "executionInfo": {
     "elapsed": 18097,
     "status": "ok",
     "timestamp": 1601699603132,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "yl53qt9DNA0E",
    "outputId": "fb4eb934-2cee-427b-abd5-87b134181a9f"
   },
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "credit = pd.read_csv(\"D:/ASU Classes/CIS 508/credit.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>checking_balance</th>\n",
       "      <th>months_loan_duration</th>\n",
       "      <th>credit_history</th>\n",
       "      <th>purpose</th>\n",
       "      <th>amount</th>\n",
       "      <th>savings_balance</th>\n",
       "      <th>employment_duration</th>\n",
       "      <th>percent_of_income</th>\n",
       "      <th>years_at_residence</th>\n",
       "      <th>age</th>\n",
       "      <th>other_credit</th>\n",
       "      <th>housing</th>\n",
       "      <th>existing_loans_count</th>\n",
       "      <th>job</th>\n",
       "      <th>dependents</th>\n",
       "      <th>phone</th>\n",
       "      <th>default</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>A11</td>\n",
       "      <td>6</td>\n",
       "      <td>A34</td>\n",
       "      <td>A43</td>\n",
       "      <td>1169</td>\n",
       "      <td>A65</td>\n",
       "      <td>A75</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>67</td>\n",
       "      <td>A143</td>\n",
       "      <td>A152</td>\n",
       "      <td>2</td>\n",
       "      <td>A173</td>\n",
       "      <td>1</td>\n",
       "      <td>A192</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A12</td>\n",
       "      <td>48</td>\n",
       "      <td>A32</td>\n",
       "      <td>A43</td>\n",
       "      <td>5951</td>\n",
       "      <td>A61</td>\n",
       "      <td>A73</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>22</td>\n",
       "      <td>A143</td>\n",
       "      <td>A152</td>\n",
       "      <td>1</td>\n",
       "      <td>A173</td>\n",
       "      <td>1</td>\n",
       "      <td>A191</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A14</td>\n",
       "      <td>12</td>\n",
       "      <td>A34</td>\n",
       "      <td>A46</td>\n",
       "      <td>2096</td>\n",
       "      <td>A61</td>\n",
       "      <td>A74</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>49</td>\n",
       "      <td>A143</td>\n",
       "      <td>A152</td>\n",
       "      <td>1</td>\n",
       "      <td>A172</td>\n",
       "      <td>2</td>\n",
       "      <td>A191</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A11</td>\n",
       "      <td>42</td>\n",
       "      <td>A32</td>\n",
       "      <td>A42</td>\n",
       "      <td>7882</td>\n",
       "      <td>A61</td>\n",
       "      <td>A74</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>45</td>\n",
       "      <td>A143</td>\n",
       "      <td>A153</td>\n",
       "      <td>1</td>\n",
       "      <td>A173</td>\n",
       "      <td>2</td>\n",
       "      <td>A191</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A11</td>\n",
       "      <td>24</td>\n",
       "      <td>A33</td>\n",
       "      <td>A40</td>\n",
       "      <td>4870</td>\n",
       "      <td>A61</td>\n",
       "      <td>A73</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>53</td>\n",
       "      <td>A143</td>\n",
       "      <td>A153</td>\n",
       "      <td>2</td>\n",
       "      <td>A173</td>\n",
       "      <td>2</td>\n",
       "      <td>A191</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  checking_balance  months_loan_duration credit_history purpose  amount  \\\n",
       "0              A11                     6            A34     A43    1169   \n",
       "1              A12                    48            A32     A43    5951   \n",
       "2              A14                    12            A34     A46    2096   \n",
       "3              A11                    42            A32     A42    7882   \n",
       "4              A11                    24            A33     A40    4870   \n",
       "\n",
       "  savings_balance employment_duration  percent_of_income  years_at_residence  \\\n",
       "0             A65                 A75                  4                   4   \n",
       "1             A61                 A73                  2                   2   \n",
       "2             A61                 A74                  2                   3   \n",
       "3             A61                 A74                  2                   4   \n",
       "4             A61                 A73                  3                   4   \n",
       "\n",
       "   age other_credit housing  existing_loans_count   job  dependents phone  \\\n",
       "0   67         A143    A152                     2  A173           1  A192   \n",
       "1   22         A143    A152                     1  A173           1  A191   \n",
       "2   49         A143    A152                     1  A172           2  A191   \n",
       "3   45         A143    A153                     1  A173           2  A191   \n",
       "4   53         A143    A153                     2  A173           2  A191   \n",
       "\n",
       "  default  \n",
       "0      no  \n",
       "1     yes  \n",
       "2      no  \n",
       "3      no  \n",
       "4     yes  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "credit.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 17)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "credit.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Then you need to perform label encoding for predictors.\n",
    "\n",
    "##### code for reference:\n",
    "\n",
    "X = pd.get_dummies(credit.iloc[ ,] , drop_first = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "executionInfo": {
     "elapsed": 363,
     "status": "ok",
     "timestamp": 1601699605388,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "OlVGcW7FNUHa"
   },
   "outputs": [],
   "source": [
    "X = pd.get_dummies(credit.iloc[ :,0:16] , drop_first = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3. Then let's perform label encoding for the target variable.\n",
    "\n",
    "##### code for reference \n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "labelencoder_credit = LabelEncoder()\n",
    "\n",
    "y = labelencoder_credit.fit_transform(credit['target variable name here'].values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "executionInfo": {
     "elapsed": 821,
     "status": "ok",
     "timestamp": 1601699607073,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "0KyvNI3WNOHb"
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "labelencoder_credit = LabelEncoder()\n",
    "\n",
    "y = labelencoder_credit.fit_transform(credit['default'].values)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "4. Then let's split the data into training set and test set using 20% of the data (using random_state = 0).\n",
    "\n",
    "##### code for reference\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "executionInfo": {
     "elapsed": 347,
     "status": "ok",
     "timestamp": 1601699610384,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "fgT33wsQNdcq"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Build a Decision Tree\n",
    "\n",
    "#### Q2. Perform a single decision tree model to predict customers’ default status. What is the performance accuracy of the decision tree model on the test set?\n",
    "\n",
    "To answer this question, you will create a single decision tree model with default parameter values using **DecisionTreeClassifier** (set **random_state= 0**), and evaluate the decision tree model on the test set.\n",
    "\n",
    "\n",
    "1.  first we build a decision tree model and apply it on the test set using **predict()**. We get the accuracy on the test set using **accuracy_score()**.\n",
    "\n",
    "##### code for reference\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "tree = DecisionTreeClassifier(random_state = 0)\n",
    "\n",
    "tree.fit(X_train, y_train)\n",
    "\n",
    "y_pred = tree.predict(X_test)\n",
    "\n",
    "print(\"Accuracy on test set: {:.3f}\".format(accuracy_score(y_pred, y_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "executionInfo": {
     "elapsed": 970,
     "status": "ok",
     "timestamp": 1601699616190,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "vgoSCJwjDcNm"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on test set: 0.680\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "tree = DecisionTreeClassifier(random_state = 0)\n",
    "\n",
    "tree.fit(X_train, y_train)\n",
    "\n",
    "y_pred = tree.predict(X_test)\n",
    "\n",
    "print(\"Accuracy on test set: {:.3f}\".format(accuracy_score(y_pred, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. we also evaluate the model performance on the test set using ROC curve with **plot_roc_curve()**.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "from sklearn import metrics \n",
    "\n",
    "from matplotlib import pyplot as plt\n",
    "\n",
    "metrics.plot_roc_curve(tree, X_test, y_test) \n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\apere\\anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function :func:`plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: :meth:`sklearn.metric.RocCurveDisplay.from_predictions` or :meth:`sklearn.metric.RocCurveDisplay.from_estimator`.\n",
      "  warnings.warn(msg, category=FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "from matplotlib import pyplot as plt\n",
    "\n",
    "metrics.plot_roc_curve(tree, X_test, y_test)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oYP4N8j2Qn3I"
   },
   "source": [
    "## Step 2: Cross Validation\n",
    "\n",
    "#### Q3. What is the model accuracy of the decision tree model after performing the cross validation?\n",
    "\n",
    "To answer this question, you need to performance cross validation using **cross_val_score**. Let's say we perform 5 fold cross validation.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "\n",
    "scores = cross_val_score(tree, X, y, cv= 5)\n",
    "\n",
    "print(\"Accuracy scores of each fold: {}\".format(scores))\n",
    "\n",
    "print(\"Average cross-validation score: {:.2f}\".format(scores.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 69
    },
    "executionInfo": {
     "elapsed": 376,
     "status": "ok",
     "timestamp": 1601531337323,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "vpBfFqYCQnHl",
    "outputId": "782946ae-8c18-4555-ef53-1cd85db40fe8"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy scores of each fold: [0.7   0.725 0.615 0.66  0.68 ]\n",
      "Average cross-validation score: 0.68\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "scores = cross_val_score(tree, X, y, cv= 5)\n",
    "\n",
    "print(\"Accuracy scores of each fold: {}\".format(scores))\n",
    "\n",
    "print(\"Average cross-validation score: {:.2f}\".format(scores.mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MrUwXFLKVF7v"
   },
   "source": [
    "## Step 4: Emsemble Methods\n",
    "In this step you will perform serveral emsemble methods and improve the model performance.\n",
    "\n",
    "### Emsemble Method 1: Bagging \n",
    "\n",
    "#### Q4. Develop a bagging model (with 100 decision trees) on the training set and evaluate the model using the test set. What is the model performance? \n",
    "\n",
    "To answer this question, you need to create a bagging model using **BaggingClassifier** (with **n_estimator = 100**, **random_state= 0**), and apply it on the test using **.predict** method, and evaluate it using accuracy and AUC.\n",
    "\n",
    "1. First, create a bagging classifier using BaggingClassifer and apply it on the test set\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "from sklearn.ensemble import BaggingClassifier\n",
    "\n",
    "bagging = BaggingClassifier(n_estimators=100, random_state=0)\n",
    "\n",
    "bagging.fit(X_train, y_train)\n",
    "\n",
    "y_bagging_pred = bagging.predict(X_test)\n",
    "\n",
    "print(\"Bagging Model Accuracy on test set: {:.3f}\".format(accuracy_score(y_test,y_bagging_pred)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bagging Model Accuracy on test set: 0.770\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import BaggingClassifier\n",
    "\n",
    "bagging = BaggingClassifier(n_estimators=100, random_state=0)\n",
    "\n",
    "bagging.fit(X_train, y_train)\n",
    "\n",
    "y_bagging_pred = bagging.predict(X_test)\n",
    "\n",
    "print(\"Bagging Model Accuracy on test set: {:.3f}\".format(accuracy_score(y_test,y_bagging_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Then we get the AUC for the bagging classifer using **plot_roc_curve**.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "metrics.plot_roc_curve(bagging, X_test, y_test)  \n",
    "\n",
    "plt.show() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\apere\\anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function :func:`plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: :meth:`sklearn.metric.RocCurveDisplay.from_predictions` or :meth:`sklearn.metric.RocCurveDisplay.from_estimator`.\n",
      "  warnings.warn(msg, category=FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics.plot_roc_curve(bagging, X_test, y_test)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Emsemble Method 2: Boosting\n",
    "\n",
    "#### Q5. Develop a boosting model (with 100 decision trees) on the training set and evaluate the model using the test set. What is the model performance? \n",
    "\n",
    "To answer this question, you need to create a boosting model using **AdaBoostClassifier** (with **n_estimator = 100**,**random_state= 0**), and apply it on the test using **.predict** method, and evaluate it using accuracy and AUC.\n",
    "\n",
    "1. First, create a boosting classifier using AdaBoostClassifier() and apply it on the test set\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "\n",
    "boost = AdaBoostClassifier(n_estimators=100, random_state=0)\n",
    "\n",
    "boost.fit(X_train, y_train)\n",
    "\n",
    "y_boost_pred = boost.predict(X_test)\n",
    "\n",
    "print(\"Accuracy on test set: {:.3f}\".format(accuracy_score(y_boost_pred, y_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on test set: 0.770\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "\n",
    "boost = AdaBoostClassifier(n_estimators=100, random_state=0)\n",
    "\n",
    "boost.fit(X_train, y_train)\n",
    "\n",
    "y_boost_pred = boost.predict(X_test)\n",
    "\n",
    "print(\"Accuracy on test set: {:.3f}\".format(accuracy_score(y_boost_pred, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Then we get the AUC for the boosting classifer using **plot_roc_curve**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\apere\\anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function :func:`plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: :meth:`sklearn.metric.RocCurveDisplay.from_predictions` or :meth:`sklearn.metric.RocCurveDisplay.from_estimator`.\n",
      "  warnings.warn(msg, category=FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics.plot_roc_curve(boost, X_test, y_test)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Emsemble Method 3: Random Forest\n",
    "\n",
    "#### Q6. Develop a random forest model (with 100 decision trees) on the training set and evaluate the model using the test set. What is the model performance? \n",
    "\n",
    "To answer this question, you need to create a random forest model using **RandomForestClassifier** (with **n_estimator = 1000**, **random_state= 0**), and apply it on the test using **.predict** method, and evaluate it using accuracy and AUC.\n",
    "\n",
    "1. First, create a random forest classifier using RandomForestClassifier() and apply it on the test set.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "forest = RandomForestClassifier(n_estimators=1000, random_state=0)\n",
    "\n",
    "forest.fit(X_train, y_train)\n",
    "\n",
    "y_rf_pred = forest.predict(X_test)\n",
    "\n",
    "print(\"Random Forest Accuracy on test set: {:.3f}\".format(accuracy_score(y_test, y_rf_pred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 52
    },
    "executionInfo": {
     "elapsed": 385,
     "status": "ok",
     "timestamp": 1601527368397,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "eUqVrtfCQIhz",
    "outputId": "4c9a5aec-f98d-4866-b505-d315d96d8141"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy on test set: 0.765\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "forest = RandomForestClassifier(n_estimators=1000, random_state=0)\n",
    "\n",
    "forest.fit(X_train, y_train)\n",
    "\n",
    "y_rf_pred = forest.predict(X_test)\n",
    "\n",
    "print(\"Random Forest Accuracy on test set: {:.3f}\".format(accuracy_score(y_test, y_rf_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Then we get the AUC for the random forest classifer using **plot_roc_curve**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\apere\\anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function :func:`plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: :meth:`sklearn.metric.RocCurveDisplay.from_predictions` or :meth:`sklearn.metric.RocCurveDisplay.from_estimator`.\n",
      "  warnings.warn(msg, category=FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAGwCAYAAABVdURTAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABaW0lEQVR4nO3deVhUZf8G8HvYhh1SdkUWDcVdwQ1fc8nd3CrFLRW3SHOj9NXXCrWSVkVzTVHK3MqtetOUSgW3FIRc8FVTBFRQUVlklZnn94c/J0cWZw4DA8P9ua65cp6z3ecMMV+e85xzZEIIASIiIiIDYaTvAERERES6xOKGiIiIDAqLGyIiIjIoLG6IiIjIoLC4ISIiIoPC4oaIiIgMCosbIiIiMigm+g5Q1ZRKJW7dugUbGxvIZDJ9xyEiIiINCCGQk5MDNzc3GBmV3zdT64qbW7duwd3dXd8xiIiISILU1FTUr1+/3HlqXXFjY2MD4PHBsbW11XMaIiIi0kR2djbc3d1V3+PlqXXFzZNTUba2tixuiIiIahhNhpRwQDEREREZFBY3REREZFBY3BAREZFBYXFDREREBoXFDRERERkUFjdERERkUFjcEBERkUFhcUNEREQGhcUNERERGRQWN0RERGRQ9FrcREdHY+DAgXBzc4NMJsPevXufu8yRI0fg5+cHc3NzeHt7Y+3atZUflIiIiGoMvRY3ubm5aNWqFVauXKnR/ElJSejfvz+6dOmC+Ph4/Oc//8GMGTOwa9euSk5KRERENYVeH5zZr18/9OvXT+P5165diwYNGiA8PBwA4Ovri9jYWHzxxRd47bXXKiklERFR9ZCZV4SHhcX6jvFcxkYyuNpZ6G37Neqp4CdOnEDv3r3V2vr06YOIiAg8evQIpqamJZYpLCxEYWGh6n12dnal5yQiItK16Mt3ERR5Ggql0HeU53KykePUgp56236NKm7S09Ph7Oys1ubs7Izi4mJkZGTA1dW1xDJhYWFYtGhRVUUkIiKqFBduZUOhFDCSAabG1ft6ILmpfvPVqOIGAGQymdp7IUSp7U/Mnz8fISEhqvfZ2dlwd3evvIBERESV6LW29fH5sFb6jlGt1ajixsXFBenp6Wptd+7cgYmJCerWrVvqMnK5HHK5vCriERERUTVQvfu1ntGpUydERUWptR08eBD+/v6ljrchIiKi2kevPTcPHz7E33//rXqflJSEhIQE1KlTBw0aNMD8+fNx8+ZNfPvttwCA4OBgrFy5EiEhIZg8eTJOnDiBiIgIbNu2TV+7QEREOiaEwJmUTNzOLtB3lGrlUjoviNGUXoub2NhYdO/eXfX+ydiYcePGITIyEmlpaUhJSVFN9/Lywr59+zB79mysWrUKbm5uWLFiBS8DJyIyIGdSMvHamuP6jlFtmRiXPsaU/qHX4qZbt26qAcGliYyMLNHWtWtXnDlzphJTERGRPt35/x4bG7kJfF1t9ZymepGbGmFEuwb6jlHt1agBxUREVHv4utri++BO+o5BNVCNGlBMRERE9DwsboiIiMigsLghIiIig8LihoiIqpW/7zwEAJiZ8CuKpOFPDhERVRv3c4vwdfQ1AMCrbevpOQ3VVCxuiIio2vjqjyvIKSxGU1dbDGnN4oakYXFDRETVQvK9XHx3MhkA8J/+vjAy4s3qSBoWN0REVC18duASHikEXvJxxL9edNB3HKrBeBM/IiLSi6NXMrAu+ioUSgGlEDh57T5kMmB+vyb6jkY1HIsbIiKqcnlFxXjnhwTczi5Uax/mV5+PXKAKY3FDRERVbuPRJNzOLkT9Fywwt+/jnhozYyN0a+yo52RkCFjcEBFRlcp4WIi1Rx5f7j2nT2MMauWm50RkaDigmIiIqtSK36/gYWExWtSzw8CWLGxI91jcEBFRlbl29yG2/pkCAJjfvwkv96ZKweKGiIiqzOcHLqFYKdCjiRMCGvJyb6ocLG6IiKhKxCXfx/7z6TCSAf/uy8u9qfKwuCEiokonhMCSff8DAAzzc0djFxs9JyJDxuKGiIgq3YELtxGX/ADmpkYI6e2j7zhk4FjcEBFRpXqkUOLTXx/32kzu4g1nW3M9JyJDx/vcEBHVMjcz81HwSFFl2ztwIR1JGbmoa2WGKS95V9l2qfZicUNEVIt8c/w6Qn+6oJdtz+r5ImzMTfWybapdWNwQEdUiF9OyAQByEyOYmxpX2XZbudtjRPsGVbY9qt1Y3BAR1ULTezTC2z1e1HcMokrBAcVERERkUFjcEBERkUFhcUNEREQGhcUNERERGRQWN0RERGRQWNwQERGRQWFxQ0RERAaFxQ0REREZFN7Ej4iohipWKBFxNAlpWQUaLxOX/KASExFVDyxuiIhqqD+T7iNs//8kLWst569/Mlz86SYiqqHyih4/2dvZVo5hfu4aL2drYYLX/OpXViwivWNxQ0RUw7nZW+DdPo31HYOo2uCAYiIiIjIoLG6IiIjIoPC0FBHRc6Tez8OCvefxsOCRvqOoycyvXnmIqguti5usrCzs2bMHMTExuH79OvLy8uDo6Ig2bdqgT58+CAgIqIycRER6s/i/iYi+fFffMcrkameu7whE1YrGxU1aWho++OADbNmyBS4uLmjfvj1at24NCwsL3L9/H4cOHcIXX3wBDw8PhIaGIjAwsDJzExFViVNJ9xGVeBvGRjJ89lpL2JhXrw5vYyMZOnjX1XcMompF4/9LW7VqhbFjx+LUqVNo3rx5qfPk5+dj7969WLp0KVJTU/Huu+/qLCgRUVUTQmDJvosAgMB27rx8mqiG0Li4uXDhAhwdHcudx8LCAiNHjsTIkSNx92717cIlItLE/vPpSEjNhKWZMWb1fFHfcYhIQxpfLfW8wqai8xMRVSdFxUp8+uvju/9OeckbTjYc10JUU+j05PGDBw/w888/Y+zYsbpcLRGR1hJSMzEx8jSyJF5RJAAolAKONnJM7uKt23BEVKl0ep+blJQUBAUF6XKVRESSnLh6D/dyi1CsFJJeCqUAAMzv1wRWfA4TUY2i1f+x2dnZ5U7PycmpUBgiIl0b0NIVH7zSVNKy5ibGsLM01XEiIqpsWhU39vb2kMlkZU4XQpQ7nYioqlmYGsPZluNliGoTrYobGxsbLFiwAB06dCh1+pUrV/Dmm2/qJBgRERGRFFoVN23btgUAdO3atdTp9vb2EEJUPBUREYDb2QXIeFgoeVkiqp20Km5GjRqF/Pz8Mqe7uLggNDS0wqGIiM7fzMKglUehrODfSzxRTlT7aFXcTJ48udzpzs7OLG6ISCeSMnKhFICpsQx1rMwkrcPc1BgDWrrqOBkRVXe8vpGIqjV/jzrYNqWjvmMQUQ2i0/vcEBEREekbixsiIiIyKCxuiIiIyKCwuCEiIiKDovfiZvXq1fDy8oK5uTn8/PwQExNT7vxbtmxBq1atYGlpCVdXVwQFBeHevXtVlJaIiIiqO8nFTffu3TF+/Hi1tnHjxqFHjx4ar2PHjh2YNWsWFixYgPj4eHTp0gX9+vVDSkpKqfMfPXoUY8eOxcSJE3HhwgX88MMPOH36NCZNmiR1N4iIiMjASC5uPD094ebmptZWr149eHh4aLyOpUuXYuLEiZg0aRJ8fX0RHh4Od3d3rFmzptT5T548CU9PT8yYMQNeXl7417/+hTfffBOxsbFlbqOwsBDZ2dlqLyIiIjJckoubTZs2YcmSJWptS5YswaZNmzRavqioCHFxcejdu7dae+/evXH8+PFSlwkICMCNGzewb98+CCFw+/Zt7Ny5EwMGDChzO2FhYbCzs1O93N3dNcpHRERENZPextxkZGRAoVDA2dlZrd3Z2Rnp6emlLhMQEIAtW7YgMDAQZmZmcHFxgb29Pb766qsytzN//nxkZWWpXqmpqTrdDyIiIqpeNL5D8YoVKzRe6YwZMzSeVyZTf/KLEKJE2xOJiYmYMWMGPvjgA/Tp0wdpaWmYM2cOgoODERERUeoycrkccrlc4zxERERUs2lc3Cxbtkyj+WQymUbFjYODA4yNjUv00ty5c6dEb84TYWFh6Ny5M+bMmQMAaNmyJaysrNClSxd89NFHcHXlM2SIiIhqO42Lm6SkJJ1u2MzMDH5+foiKisLQoUNV7VFRURg8eHCpy+Tl5cHERD2ysbExgMc9PkREREQVGnNTVFSES5cuobi4WNLyISEh2LBhAzZu3IiLFy9i9uzZSElJQXBwMIDH42XGjh2rmn/gwIHYvXs31qxZg2vXruHYsWOYMWMG2rdvX+LKLSIiIqqdJD0VPC8vD9OnT8c333wDALh8+TK8vb0xY8YMuLm5Yd68eRqtJzAwEPfu3cPixYuRlpaG5s2bY9++farLydPS0tTueTN+/Hjk5ORg5cqVeOedd2Bvb48ePXrg008/lbIbREREZIBkQsL5nJkzZ+LYsWMIDw9H3759cfbsWXh7e+Onn35CaGgo4uPjKyOrTmRnZ8POzg5ZWVmwtbXVdxyiGunX8+lYffhvFCsq73RwVv4j3MzMRyfvutg2pWOlbYeIagZtvr8l9dzs3bsXO3bsQMeOHdWubGratCmuXr0qZZVEVINsPnkdZ29kVcm26r1gUSXbISLDIam4uXv3LpycnEq05+bmlnkZNxEZDqXy8X+ndmuIjt51K207JkYy+Hm+UGnrJyLDJKm4adeuHX755RdMnz4dwD/3qlm/fj06deqku3REVK35utriJR9HfccgIlIjqbgJCwtD3759kZiYiOLiYixfvhwXLlzAiRMncOTIEV1nJCIiItKYpEvBAwICcOzYMeTl5aFhw4Y4ePAgnJ2dceLECfj5+ek6IxFVQLFCqfOXkveVIqJqTFLPDQC0aNFCdSk4EVU/CqXAmA1/4sS1e/qOQkRUpSQXNwqFAnv27MHFixchk8ng6+uLwYMHl7iDMBHpx664G5Va2NjITdDMjbdTIKLqR1Ilcv78eQwePBjp6elo3LgxgMc38nN0dMRPP/2EFi1a6DQkEWknv0iBL6MuAQDm9GmMMR08dL4NczMjyE2Mdb5eIqKKklTcTJo0Cc2aNUNsbCxeeOHxZZoPHjzA+PHjMWXKFJw4cUKnIYlIOxuPJeF2diHq2Vtg4r+8YG7KIoSIag9Jxc1ff/2lVtgAwAsvvICPP/4Y7dq101k4ItLevYeFWHP48c005/ZtzMKGiGodScVN48aNcfv2bTRr1kyt/c6dO2jUqJFOghGROqVS4MKtbBQplOXOt+1UCh4WFqN5PVsMbMkHyhJR7aNxcZOdna3695IlSzBjxgwsXLgQHTs+fubLyZMnsXjxYj7EkqiSfH7wkqpHRhP/6ecLIyPeMZyIah+Nixt7e3u1RysIITB8+HBV25Pnbw4cOBAKhULHMYnoekYuAKCOlRlszMv/X7enrzMCGjlURSwiompH4+Lm0KFDlZmDiDQ0u5cP3uio+6ufiIgMhcbFTdeuXSszBxEREZFOVOiOe3l5eUhJSUFRUZFae8uWLSsUioiIiEgqScXN3bt3ERQUhP3795c6nWNuiKQ7eyMTcckPSrQn/f+YGyIiKp+k4mbWrFl48OABTp48ie7du2PPnj24ffs2PvroI3z55Ze6zkhUayiUAqM3/ImcguIy55GbSHreLRFRrSGpuPnjjz/w448/ol27djAyMoKHhwd69eoFW1tbhIWFYcCAAbrOSVQrKJRCVdj0beYC02cKmbpWZujT1EUf0YiIagxJxU1ubi6cnJwAAHXq1MHdu3fh4+ODFi1a4MyZMzoNSFRbfTasJWzNTfUdg4ioxpHUv924cWNcuvT4oXytW7fGunXrcPPmTaxduxaurq46DUhERESkDcljbtLS0gAAoaGh6NOnD7Zs2QIzMzNERkbqMh8RERGRViQVN6NHj1b9u02bNrh+/Tr+97//oUGDBnBw4F1RibQhhMCqQ3/j6t1cKJRC33GIiGq8Ct3n5glLS0u0bdtWF6siqnWu3HmILw5eVmuzMDWGmTGviiIikkLj4iYkJETjlS5dulRSGKLaqPDR46d825ibYObLLwIAWrvbw9zUWJ+xiIhqLI2Lm/j4eI3me/rhmkSkORu5CSZ18dZ3DCKiGo8PziQiIiKDwpP6REREZFB0MqCYyJDsPnMDaw5fhUJUzZVLT8bcEBGRbrC4IXrGlj9TcOXOwyrfboO6llW+TSIiQ8TihugZyv/vsZnTpzHaedapkm3KZEBzN7sq2RYRkaFjcUNUBh9nG7T3qprihoiIdEfygOLNmzejc+fOcHNzQ3JyMgAgPDwcP/74o87CEREREWlLUnGzZs0ahISEoH///sjMzIRCoQAA2NvbIzw8XJf5iIiIiLQiqbj56quvsH79eixYsADGxv/cRdXf3x/nzp3TWTgiIiIibUkqbpKSktCmTZsS7XK5HLm5uRUORURERCSVpOLGy8sLCQkJJdr379+Ppk2bVjQTERERkWSSrpaaM2cOpk2bhoKCAgghcOrUKWzbtg1hYWHYsGGDrjMSERERaUxScRMUFITi4mLMnTsXeXl5GDVqFOrVq4fly5djxIgRus5IREREpDHJ97mZPHkyJk+ejIyMDCiVSjg5OekyF9Fz3ckuwN+VcCfhnIJina+TiIiqjqTiZtGiRRgzZgwaNmwIBwcHXWcieq7MvCL0Wx6De7lFlbYNI1mlrZqIiCqRpOJm165dWLx4Mdq1a4cxY8YgMDAQjo6Ous5GVKZVh/7Gvdwi2JqbwNXOQufrd7EzRzvenZiIqEaSCSHt0ccXLlzAli1bsH37dty4cQM9e/bEmDFjMGTIEFhaVt8HAGZnZ8POzg5ZWVmwtbXVdxySIPV+Hl7+8giKFEpEBrVDt8Y8JUpEZOi0+f6W/PiFZs2aYcmSJbh27RoOHToELy8vzJo1Cy4uLlJXSaSRLw5eQpFCic6N6qKrD3sMiYhIneTi5mlWVlawsLCAmZkZHj16pItVEpXq3I0s/JhwCwAwv58vZDIOjCEiInWSr5ZKSkrC1q1bsWXLFly+fBkvvfQSFi5ciGHDhukyH9US6VkF+O/ZW3ikKP8s6f7zaQCAoW3qoXk9u6qIRkRENYyk4qZTp044deoUWrRogaCgINV9boik+mT/Rez9/x6Z5zEzMcI7vX0qOREREdVUkoqb7t27Y8OGDWjWrJmu81AtlZX/+HRmO88X4FnXqtx5ezZ1Rv0Xqu+gdSIi0i9Jxc2SJUt0nYMIADDc3x3D/N31HYOIiGowjYubkJAQfPjhh7CyskJISEi58y5durTCwYiIiIik0Li4iY+PV10JFR8fX2mBiIiIiCpC4+Lm0KFDpf6biIiIqDqRdJ+bCRMmICcnp0R7bm4uJkyYUOFQRERERFJJKm6++eYb5Ofnl2jPz8/Ht99+W+FQRERERFJpdbVUdnY2hBAQQiAnJwfm5uaqaQqFAvv27YOTE5/zQ0RERPqjVXFjb28PmUwGmUwGH5+SN1GTyWRYtGiRzsIRERERaUur01KHDh3C77//DiEEdu7ciT/++EP1Onr0KFJSUrBgwQKtAqxevRpeXl4wNzeHn58fYmJiyp2/sLAQCxYsgIeHB+RyORo2bIiNGzdqtU0iIiIyXFr13HTt2hXA4+dKNWjQoMIPLdyxYwdmzZqF1atXo3Pnzli3bh369euHxMRENGjQoNRlhg8fjtu3byMiIgKNGjXCnTt3UFxcXKEcVHEnrt7De3vPoeCRUtLyGQ8LdZyIiIhqK5kQovwnFf6/s2fPonnz5jAyMsLZs2fLnbdly5YabbxDhw5o27Yt1qxZo2rz9fXFkCFDEBYWVmL+X3/9FSNGjMC1a9dQp04djbZRWFiIwsJ/vjizs7Ph7u6OrKws2NraarQOer73957H5pPJFV7Prrc6wc9Ds8+WiIhqj+zsbNjZ2Wn0/a1xz03r1q2Rnp4OJycntG7dGjKZDKXVRTKZDAqF4rnrKyoqQlxcHObNm6fW3rt3bxw/frzUZX766Sf4+/vjs88+w+bNm2FlZYVBgwbhww8/hIWFRanLhIWFcRxQFRB4/LMwop07RrYvvdfteepam/GZUUREVGEaFzdJSUlwdHRU/buiMjIyoFAo4OzsrNbu7OyM9PT0Upe5du0ajh49CnNzc+zZswcZGRmYOnUq7t+/X+a4m/nz56s9LuJJzw1VDmdbc7Ryt9d3DCIiqsU0Lm48PDxK/XdFPTtuRwhR5lgepVIJmUyGLVu2wM7ODsDj51i9/vrrWLVqVam9N3K5HHK5XGd5iYiIqHqTfBO/X375RfV+7ty5sLe3R0BAAJKTNRt34eDgAGNj4xK9NHfu3CnRm/OEq6sr6tWrpypsgMdjdIQQuHHjhoQ9ISIiIkMjqbhZsmSJqpfkxIkTWLlyJT777DM4ODhg9uzZGq3DzMwMfn5+iIqKUmuPiopCQEBAqct07twZt27dwsOHD1Vtly9fhpGREerXry9lV4iIiMjASCpuUlNT0ahRIwDA3r178frrr2PKlCkICwt77n1qnhYSEoINGzZg48aNuHjxImbPno2UlBQEBwcDeDxeZuzYsar5R40ahbp16yIoKAiJiYmIjo7GnDlzMGHChDIHFBMREVHtotV9bp6wtrbGvXv30KBBAxw8eFDVW2Nubl7qM6fKEhgYiHv37mHx4sVIS0tD8+bNsW/fPtWYnrS0NKSkpKhtNyoqCtOnT4e/vz/q1q2L4cOH46OPPpKyG0RERGSAJBU3vXr1wqRJk9CmTRtcvnwZAwYMAABcuHABnp6eWq1r6tSpmDp1aqnTIiMjS7Q1adKkxKksIiIioicknZZatWoVOnXqhLt372LXrl2oW7cuACAuLg4jR47UaUAiIiIibUjqubG3t8fKlStLtPNmeURERKRvkoobAMjMzERERAQuXrwImUwGX19fTJw4Ue0ybSIiIqKqJum0VGxsLBo2bIhly5bh/v37yMjIwLJly9CwYUOcOXNG1xmJiIiINCap52b27NkYNGgQ1q9fDxOTx6soLi7GpEmTMGvWLERHR+s0JBEREZGmJBU3sbGxaoUNAJiYmGDu3Lnw9/fXWTgiIiIibUk6LWVra6t2/5knUlNTYWNjU+FQRERERFJJKm4CAwMxceJE7NixA6mpqbhx4wa2b9+OSZMm8VJwIiIi0itJp6W++OILyGQyjB07FsXFxQAAU1NTvPXWW/jkk090GpCIiIhIG5KKGzMzMyxfvhxhYWG4evUqhBBo1KgRLC0tdZ2PiIiISCtanZbKy8vDtGnTUK9ePTg5OWHSpElwdXVFy5YtWdgQERFRtaBVcRMaGorIyEgMGDAAI0aMQFRUFN56663KykZERESkNa1OS+3evRsREREYMWIEAGDMmDHo3LkzFAoFjI2NKyUgERERkTa06rlJTU1Fly5dVO/bt28PExMT3Lp1S+fBiIiIiKTQqrhRKBQwMzNTazMxMVFdMUVERESkb1qdlhJCYPz48ZDL5aq2goICBAcHw8rKStW2e/du3SUkIiIi0oJWxc24ceNKtI0ZM0ZnYYiIiIgqSqviZtOmTZWVg4iIiEgnJD1+gYiIiKi60ri4CQ4ORmpqqkbz7tixA1u2bJEcimoepdB3AiIiosc0Pi3l6OiI5s2bIyAgAIMGDYK/vz/c3Nxgbm6OBw8eIDExEUePHsX27dtRr149fP3115WZm6qRvKJi/JZ4GwBQz95Cz2mIiKi2kwkhNP6b+86dO4iIiMD27dtx/vx5tWk2Njbo2bMnpkyZgt69e+s8qK5kZ2fDzs4OWVlZsLW11Xccg/DV71fwZdRluNexwG8hXSE34Q0diYhIt7T5/taquHlaZmYmkpOTkZ+fDwcHBzRs2BAymUxS4KrE4ka37uYUotvnh5BbpMCKkW0wqJWbviMREZEB0ub7W9JTwQHA3t4e9vb2UhcnA7Hi9yvILVKgZX07vNLCVd9xiIiIeLUUSXf17kNsPZUCAJjfzxdGRtW/546IiAwfixuSJLvgEebuPAuFUuDlJk7o1LCuviMREREBqMBpKaq97uQUYNzG07iYlg1ruQnm9/fVdyQiIiIVFjekleR7uXgj4hRS7ufBwVqOyKB2aORkre9YREREKpJPSxUXF+O3337DunXrkJOTAwC4desWHj58qLNwVL2cv5mF19acQMr9PDSoY4ldb3VC83p2+o5FRESkRlLPTXJyMvr27YuUlBQUFhaiV69esLGxwWeffYaCggKsXbtW1zlJz05cvYfJ38biYWExfF1t8c2EdnCyMdd3LCIiohIkFTczZ86Ev78//vrrL9St+89A0qFDh2LSpEk6C0f6k5CaiVuZ+QCA9KwCfLL/fyhSKNHBqw7Wj/OHrbmpnhMSERGVTlJxc/ToURw7dgxmZmZq7R4eHrh586ZOgpH+JN7KxpBVx0q092nmjOUj2sDclHcgJiKi6ktScaNUKqFQKEq037hxAzY2NhUORfp1O6cAAGBpZozmbo/H1HRqWBczXn4RxryXDRERVXOSiptevXohPDxc9XBMmUyGhw8fIjQ0FP3799dpQNKfho7W+D64k75jEBERaUVScbNs2TJ0794dTZs2RUFBAUaNGoUrV67AwcEB27Zt03VGIiIiIo1JKm7c3NyQkJCA7du3Iy4uDkqlEhMnTsTo0aNhYWGh64xEREREGpNU3ERHRyMgIABBQUEICgpStRcXFyM6OhovvfSSzgJS1cgpeITNJ5ORlf8Iqffz9B2HiIhIMknFTffu3ZGWlgYnJye19qysLHTv3r3UwcZUve1NuIXPfr2k1mZpxquiiIio5pFU3AghIJOVvGrm3r17sLKyqnAoqnq5hcUAgBedrNHVxxHGRjIMau2m51RERETa06q4efXVVwE8vjpq/PjxkMvlqmkKhQJnz55FQECAbhNSlWrlbo/3Xmmq7xhERESSaVXc2Nk9vueJEAI2NjZqg4fNzMzQsWNHTJ48WbcJiYiIiLSgVXGzadMmAICnpyfeffddnoIiIiKiakfSmJvQ0FBd5yAiIiLSCUnFDQDs3LkT33//PVJSUlBUVKQ27cyZMxUORkRERCSFkZSFVqxYgaCgIDg5OSE+Ph7t27dH3bp1ce3aNfTr10/XGYmIiIg0Jqm4Wb16Nb7++musXLkSZmZmmDt3LqKiojBjxgxkZWXpOiMRERGRxiQVNykpKapLvi0sLJCTkwMAeOONN/hsKSIiItIrScWNi4sL7t27BwDw8PDAyZMnAQBJSUkQQuguHREREZGWJA0o7tGjB37++We0bdsWEydOxOzZs7Fz507ExsaqbvRH1d/+c2l4b+95FBYrUVSs1HccIiIinZBU3Hz99ddQKh9/GQYHB6NOnTo4evQoBg4ciODgYJ0GpMpzMPE27uWqX+nW3M1WT2mIiIh0Q1JxY2RkBCOjf85oDR8+HMOHDwcA3Lx5E/Xq1dNNOqoSU7s1RGA7d5ibGsPZ1lzfcYiIiCpE0pib0qSnp2P69Olo1KiRrlZJVaSOlRk86lqxsCEiIoOgVXGTmZmJ0aNHw9HREW5ublixYgWUSiU++OADeHt74+TJk9i4cWNlZSUiIiJ6Lq1OS/3nP/9BdHQ0xo0bh19//RWzZ8/Gr7/+ioKCAuzfvx9du3atrJxEREREGtGquPnll1+wadMm9OzZE1OnTkWjRo3g4+OD8PDwSopHREREpB2tTkvdunULTZs2BQB4e3vD3NwckyZNqpRgRERERFJoVdwolUqYmpqq3hsbG8PKyqpCAVavXg0vLy+Ym5vDz88PMTExGi137NgxmJiYoHXr1hXaPhERERkWrU5LCSEwfvx4yOVyAEBBQQGCg4NLFDi7d+/WaH07duzArFmzsHr1anTu3Bnr1q1Dv379kJiYiAYNGpS5XFZWFsaOHYuXX34Zt2/f1mYXiIiIyMBp1XMzbtw4ODk5wc7ODnZ2dhgzZgzc3NxU75+8NLV06VJMnDgRkyZNgq+vL8LDw+Hu7o41a9aUu9ybb76JUaNGoVOnTtrEJyIiolpAq56bTZs26WzDRUVFiIuLw7x589Tae/fujePHj5eb4erVq/juu+/w0UcfPXc7hYWFKCwsVL3Pzs6WHroaE0LgxNV7uJ1ToPEyKffzKjERERGRfki6Q7EuZGRkQKFQwNnZWa3d2dkZ6enppS5z5coVzJs3DzExMTAx0Sx6WFgYFi1aVOG81d2fSfcxasOfkpY1MZLpOA0REZH+6K24eUImU/9iFUKUaAMAhUKBUaNGYdGiRfDx8dF4/fPnz0dISIjqfXZ2Ntzd3aUHrqbu5jzunbKzMEXL+pqfGnzB0gz9WrhWViwiIqIqp7fixsHBAcbGxiV6ae7cuVOiNwcAcnJyEBsbi/j4eLz99tsAHl+9JYSAiYkJDh48iB49epRYTi6XqwZA1wZNXW2xeWIHfccgIiLSG509W0pbZmZm8PPzQ1RUlFp7VFQUAgICSsxva2uLc+fOISEhQfUKDg5G48aNkZCQgA4d+IVOREREej4tFRISgjfeeAP+/v7o1KkTvv76a6SkpCA4OBjA41NKN2/exLfffgsjIyM0b95cbXknJyeYm5uXaCciIqLaS3LPzebNm9G5c2e4ubkhOTkZABAeHo4ff/xR43UEBgYiPDwcixcvRuvWrREdHY19+/bBw8MDAJCWloaUlBSpEYmIiKgWklTcrFmzBiEhIejfvz8yMzOhUCgAAPb29lo/Z2rq1Km4fv06CgsLERcXh5deekk1LTIyEocPHy5z2YULFyIhIUHCHhAREZGhklTcfPXVV1i/fj0WLFgAY2NjVbu/vz/OnTuns3BERERE2pJU3CQlJaFNmzYl2uVyOXJzcyscioiIiEgqScWNl5dXqaeD9u/fr3pqOBEREZE+SLpaas6cOZg2bRoKCgoghMCpU6ewbds2hIWFYcOGDbrOSERERKQxScVNUFAQiouLMXfuXOTl5WHUqFGoV68eli9fjhEjRug6IxEREZHGJN/nZvLkyZg8eTIyMjKgVCrh5OSky1xEREREkkgac7No0SJcvXoVwOPHKLCwISIioupCUnGza9cu+Pj4oGPHjli5ciXu3r2r61xEREREkkgqbs6ePYuzZ8+iR48eWLp0KerVq4f+/ftj69atyMvL03VGIiIiIo1JfvxCs2bNsGTJEly7dg2HDh2Cl5cXZs2aBRcXF13mIyIiItKKTp4KbmVlBQsLC5iZmeHRo0e6WCURERGRJJKLm6SkJHz88cdo2rQp/P39cebMGSxcuBDp6em6zEdERESkFUmXgnfq1AmnTp1CixYtEBQUpLrPDREREZG+SSpuunfvjg0bNqBZs2a6zkNERERUIZKKmyVLlug6BxEREZFOaFzchISE4MMPP4SVlRVCQkLKnXfp0qUVDkaaSbmXh5zCR7jxIF/fUYiIiKoFjYub+Ph41ZVQ8fHxlRaINLc3/iZm7UhQa5PJ9JOFiIioutC4uDl06FCp/yb9+fvOQwCAhakxbC1MYGJkhFfb1tdzKiIiIv2SdCn4hAkTkJOTU6I9NzcXEyZMqHAo0k5gO3f8+Z+eODavB173Y3FDRES1m6Ti5ptvvkF+fskxHvn5+fj2228rHIqIiIhIKq2ulsrOzoYQAkII5OTkwNzcXDVNoVBg3759fEI4ERER6ZVWxY29vT1kMhlkMhl8fHxKTJfJZFi0aJHOwhERERFpS6vi5tChQxBCoEePHti1axfq1KmjmmZmZgYPDw+4ubnpPCQRERGRprQqbrp27Qrg8XOlGjRoABmvOyYiIqJqRuPi5uzZs2jevDmMjIyQlZWFc+fOlTlvy5YtdRKOiIiISFsaFzetW7dGeno6nJyc0Lp1a8hkMgghSswnk8mgUCh0GpKIiIhIUxoXN0lJSXB0dFT9m4iIiKg60ri48fDwKPXfRERERNWJ5Jv4/fLLL6r3c+fOhb29PQICApCcnKyzcERERETaklTcLFmyBBYWFgCAEydOYOXKlfjss8/g4OCA2bNn6zQgERERkTa0uhT8idTUVDRq1AgAsHfvXrz++uuYMmUKOnfujG7duukyHxEREZFWJPXcWFtb4969ewCAgwcPomfPngAAc3PzUp85RURERFRVJPXc9OrVC5MmTUKbNm1w+fJlDBgwAABw4cIFeHp66jIfERERkVYkFTerVq3Ce++9h9TUVOzatQt169YFAMTFxWHkyJE6DUjqDl+6g2W/XcGjYiXu5BToOw4REVG1I6m4sbe3x8qVK0u086GZlW/rnyn4KzVTra2evYV+whAREVVDkoobAMjMzERERAQuXrwImUwGX19fTJw4EXZ2drrMR89Q/v9NoYM6e6J7YydYyY3Rxv0F/YYiIiKqRiQNKI6NjUXDhg2xbNky3L9/HxkZGVi2bBkaNmyIM2fO6DojlcLH2QYv+TjCz6MOjIz4AFMiIqInJPXczJ49G4MGDcL69ethYvJ4FcXFxZg0aRJmzZqF6OhonYYkIiIi0pSk4iY2NlatsAEAExMTzJ07F/7+/joLR/8oVigBoNSHlRIREdE/JBU3tra2SElJQZMmTdTaU1NTYWNjo5Ng9I+FP11A5PHr+o5BRERUI0gacxMYGIiJEydix44dSE1NxY0bN7B9+3ZMmjSJl4JXgt//d1vtvaWZMVrU48BtIiKi0kjqufniiy8gk8kwduxYFBcXAwBMTU3x1ltv4ZNPPtFpQPrHtxPao1V9e8hNjWBuaqzvOERERNWSpOLGzMwMy5cvR1hYGK5evQohBBo1agRLS0td56OnWJubwM7SVN8xiIiIqjWtTkvl5eVh2rRpqFevHpycnDBp0iS4urqiZcuWLGyIiIioWtCquAkNDUVkZCQGDBiAESNGICoqCm+99VZlZSMiIiLSmlanpXbv3o2IiAiMGDECADBmzBh07twZCoUCxsYcA0JERET6p1XPTWpqKrp06aJ63759e5iYmODWrVs6D0ZEREQkhVbFjUKhgJmZmVqbiYmJ6oopIiIiIn3T6rSUEALjx4+HXC5XtRUUFCA4OBhWVlaqtt27d+suIREREZEWtCpuxo0bV6JtzJgxOgtDREREVFFaFTebNm2qrBz0jITUTMSnPAAA5BTwtB8REZGmJN3EjyrXI4USo9efRG6RQq1dbiLpaRlERES1CoubaqhYIVSFTf8WLjA2MoJXXUs0dbXVczIiIqLqj8VNNffFsFawNOPHREREpCme5yAiIiKDwuKGiIiIDIrk4mbz5s3o3Lkz3NzckJycDAAIDw/Hjz/+qNV6Vq9eDS8vL5ibm8PPzw8xMTFlzrt792706tULjo6OsLW1RadOnXDgwAGpu0BEREQGSFJxs2bNGoSEhKB///7IzMyEQvF48Ku9vT3Cw8M1Xs+OHTswa9YsLFiwAPHx8ejSpQv69euHlJSUUuePjo5Gr169sG/fPsTFxaF79+4YOHAg4uPjpewGERERGSCZEEJou1DTpk2xZMkSDBkyBDY2Nvjrr7/g7e2N8+fPo1u3bsjIyNBoPR06dEDbtm2xZs0aVZuvry+GDBmCsLAwjdbRrFkzBAYG4oMPPtBo/uzsbNjZ2SErKwu2ttXz6qP8IgV8P/gVAJC4uA8HFBMRUa2nzfe3pJ6bpKQktGnTpkS7XC5Hbm6uRusoKipCXFwcevfurdbeu3dvHD9+XKN1KJVK5OTkoE6dOmXOU1hYiOzsbLUXERERGS5JxY2XlxcSEhJKtO/fvx9NmzbVaB0ZGRlQKBRwdnZWa3d2dkZ6erpG6/jyyy+Rm5uL4cOHlzlPWFgY7OzsVC93d3eN1k1EREQ1k6TzHXPmzMG0adNQUFAAIQROnTqFbdu2ISwsDBs2bNBqXTKZTO29EKJEW2m2bduGhQsX4scff4STk1OZ882fPx8hISGq99nZ2SxwiIiIDJik4iYoKAjFxcWYO3cu8vLyMGrUKNSrVw/Lly/HiBEjNFqHg4MDjI2NS/TS3Llzp0RvzrN27NiBiRMn4ocffkDPnj3LnVcul6s9xZyIiIgMm+RLwSdPnozk5GTcuXMH6enpSE1NxcSJEzVe3szMDH5+foiKilJrj4qKQkBAQJnLbdu2DePHj8fWrVsxYMAAqfGJiIjIQFX4MhwHBwfJy4aEhOCNN96Av78/OnXqhK+//hopKSkIDg4G8PiU0s2bN/Htt98CeFzYjB07FsuXL0fHjh1VvT4WFhaws7Or6K4QERGRAZBU3Hh5eZU7LubatWsarScwMBD37t3D4sWLkZaWhubNm2Pfvn3w8PAAAKSlpand82bdunUoLi7GtGnTMG3aNFX7uHHjEBkZKWVXiIiIyMBIKm5mzZql9v7Ro0eIj4/Hr7/+ijlz5mi1rqlTp2Lq1KmlTnu2YDl8+LBW6yYiIqLaR1JxM3PmzFLbV61ahdjY2AoFqs2KFUoUFCuRX6TQdxQiIqIaS6e3vu3Xrx/mz5+PTZs26XK1tUJ2wSP0XhqN9OwCfUchIiKq0XT6VPCdO3eWe7dgKtvVOw9LFDbtPevAwtRYT4mIiIhqJkk9N23atFEbUCyEQHp6Ou7evYvVq1frLFxtVM/eAr+/0xUAIDcx0uiGhkRERPQPScXNkCFD1N4bGRnB0dER3bp1Q5MmTXSRq9YyMgLM2VtDREQkmdbFTXFxMTw9PdGnTx+4uLhURiYiIiIiybQec2NiYoK33noLhYWFlZGHiIiIqEIkDSju0KED4uPjdZ2FiIiIqMIkjbmZOnUq3nnnHdy4cQN+fn6wsrJSm96yZUudhCMiIiLSllbFzYQJExAeHo7AwEAAwIwZM1TTZDIZhBCQyWRQKHgTOiIiItIPrYqbb775Bp988gmSkpIqKw8RERFRhWhV3AghAED1YEsiIiKi6kbrAcW8qRwRERFVZ1oPKPbx8XlugXP//n3JgYiIiIgqQuviZtGiRbCzs6uMLEREREQVpnVxM2LECDg5OVVGFiIiIqIK02rMDcfbEBERUXWnVXHz5GopIiIioupKq9NSSqWysnIYtNjr9xF5/DoUyrKLwwd5RVWYiIiIyHBJevwCaeerP/7Gkct3NZq3jpW8ktMQEREZNhY3VaCo+HGP13D/+mhRr5wrzWQydPNxrKJUREREhonFTRXq8qIjBrZy03cMIiIig6b1HYqJiIiIqjMWN0RERGRQWNwQERGRQWFxQ0RERAaFxQ0REREZFBY3REREZFBY3BAREZFB4X1udKhYoURm/qMS7Y8UfGwFERFRVWFxoyPFCiV6h0fj2t1cfUchIiKq1XhaSkfu5xWVW9g42cjR2t2+6gIRERHVUuy50TEjGXAtbIC+YxAREdVa7LkhIiIig8LihoiIiAwKixsiIiIyKCxuiIiIyKCwuCEiIiKDwuKGiIiIDAqLGyIiIjIoLG6IiIjIoLC4ISIiIoPC4oaIiIgMCosbIiIiMih8thRRDSeEQHFxMRQKhb6jEBFViKmpKYyNjSu8HhY3RDVYUVER0tLSkJeXp+8oREQVJpPJUL9+fVhbW1doPSxuiGoopVKJpKQkGBsbw83NDWZmZpDJZPqORUQkiRACd+/exY0bN/Diiy9WqAeHxQ1RDVVUVASlUgl3d3dYWlrqOw4RUYU5Ojri+vXrePToUYWKGw4oJqrhjIz4vzERGQZd9T7ztyIREREZFBY3REREZFBY3BAREZFBYXFDRLWOp6cnwsPD9R2jxhk/fjyGDBlSJdt69jNKT09Hr169YGVlBXt7ewCPx2fs3bu30jK89NJL2Lp1a6Wtv7YpLCxEgwYNEBcXV+nbYnFDRFVu/PjxkMlkkMlkMDExQYMGDfDWW2/hwYMH+o6mU56enqr9fPKqX7++3jOVVtgJIfD111+jQ4cOsLa2hr29Pfz9/REeHq6X+yidPn0aU6ZMUb1ftmwZ0tLSkJCQgMuXLwMA0tLS0K9fv0rZ/n//+1+kp6djxIgRJaYtWbIExsbG+OSTT0pMW7hwIVq3bl2iPTMzEzKZDIcPH1Zr37VrF7p16wY7OztYW1ujZcuWWLx4Me7fv6+rXSmhsLAQ06dPh4ODA6ysrDBo0CDcuHGj3GVK+1mWyWSYNm2aap7SpstkMnz++ecAALlcjnfffRf//ve/K23fnmBxQ2RAhBDIKyrWy0sIoVXWvn37Ii0tDdevX8eGDRvw888/Y+rUqZV0ZPRn8eLFSEtLU73i4+Mlr+vRo0c6TKbujTfewKxZszB48GAcOnQICQkJeP/99/Hjjz/i4MGDlbbdsjg6Oqrd4uDq1avw8/PDiy++CCcnJwCAi4sL5HK55G0UFRWVOW3FihUICgoq9WrETZs2Ye7cudi4caPkbQPAggULEBgYiHbt2mH//v04f/48vvzyS/z111/YvHlzhdZdnlmzZmHPnj3Yvn07jh49iocPH+KVV14p9y7np0+fVvs5joqKAgAMGzZMNc/T09PS0rBx40bIZDK89tprqnlGjx6NmJgYXLx4sdL2DwAgapmsrCwBQGRlZel0vbez84XHv/8rvOb9V6frJSpLfn6+SExMFPn5+aq23MJHwuPf/9XLK7fwkcbZx40bJwYPHqzWFhISIurUqaN6X1xcLCZMmCA8PT2Fubm58PHxEeHh4aWu5/PPPxcuLi6iTp06YurUqaKoqEg1z+3bt8Urr7wizM3Nhaenp/juu++Eh4eHWLZsmWqe5ORkMWjQIGFlZSVsbGzEsGHDRHp6ump6aGioaNWqlYiIiBDu7u7CyspKBAcHi+LiYvHpp58KZ2dn4ejoKD766CO1fM9u51mrV68W3t7ewtTUVPj4+Ihvv/1WbToAsWbNGjFo0CBhaWkpPvjgAyGEED/99JNo27atkMvlwsvLSyxcuFA8evTP8Q8NDRXu7u7CzMxMuLq6iunTpwshhOjatasAoPYSQogdO3YIAGLv3r0lMiqVSpGZmal2vJ/Yv3+/6Ny5s7CzsxN16tQRAwYMEH///bdqemFhoZg2bZpwcXERcrlceHh4iCVLljw357PHzsPDQy3zuHHjVMdnz549qmVu3Lghhg8fLuzt7UWdOnXEoEGDRFJSkmr6k/xLliwRrq6uwsPDo9TP5e7du0Imk4nz58+XmHb48GFRr149UVRUJNzc3MSRI0fUpj/5WXnWgwcPBABx6NAhIYQQf/75pwBQ4mf66fkrQ2ZmpjA1NRXbt29Xtd28eVMYGRmJX3/9VeP1zJw5UzRs2FAolcoy5xk8eLDo0aNHifZu3bqJ999/v9RlSvu99oQ2399677lZvXo1vLy8YG5uDj8/P8TExJQ7/5EjR+Dn5wdzc3N4e3tj7dq1VZSUiCrLtWvX8Ouvv8LU1FTVplQqUb9+fXz//fdITEzEBx98gP/85z/4/vvv1ZY9dOgQrl69ikOHDuGbb75BZGQkIiMjVdPHjx+P69ev448//sDOnTuxevVq3LlzRzVdCIEhQ4bg/v37OHLkCKKionD16lUEBgaqbefq1avYv38/fv31V2zbtg0bN27EgAEDcOPGDRw5cgSffvop3nvvPZw8eVKjfd6zZw9mzpyJd955B+fPn8ebb76JoKAgHDp0SG2+0NBQDB48GOfOncOECRNw4MABjBkzBjNmzEBiYiLWrVuHyMhIfPzxxwCAnTt3YtmyZVi3bh2uXLmCvXv3okWLFgCA3bt3o379+mq9SQCwZcsWNG7cGIMHDy6RUyaTwc7OrtR9yM3NRUhICE6fPo3ff/8dRkZGGDp0KJRKJYDHvR8//fQTvv/+e1y6dAnfffcdPD09n5vzWadPn0bfvn0xfPhwpKWlYfny5SXmycvLQ/fu3WFtbY3o6GgcPXoU1tbW6Nu3r1oPze+//46LFy8iKioK//3vf0vd3tGjR2FpaQlfX98S0yIiIjBy5EiYmppi5MiRiIiIKHUdz7NlyxZYW1uX2Vv5ZFxRaZo1awZra+syX82aNStz2bi4ODx69Ai9e/dWtbm5uaF58+Y4fvy4RtmLiorw3XffYcKECWXel+b27dv45ZdfMHHixBLT2rdv/9zv+gp7bvlTibZv3y5MTU3F+vXrRWJiopg5c6awsrISycnJpc5/7do1YWlpKWbOnCkSExPF+vXrhampqdi5c6fG22TPDRmK0v7CUSqVIrfwkV5e5f0F96xx48YJY2NjYWVlJczNzVV/kS9durTc5aZOnSpee+01tfV4eHiI4uJiVduwYcNEYGCgEEKIS5cuCQDi5MmTqukXL14UAFS9AgcPHhTGxsYiJSVFNc+FCxcEAHHq1CkhxOO/xi0tLUV2drZqnj59+ghPT0+hUChUbY0bNxZhYWGq9x4eHsLMzExYWVmpXsuXLxdCCBEQECAmT56stn/Dhg0T/fv3V70HIGbNmqU2T5cuXdR6P4QQYvPmzcLV1VUIIcSXX34pfHx81HqvnlZab5Kvr68YNGhQqfM/rbQet6fduXNHABDnzp0TQggxffp00aNHj1J/NrTNOXjwYFWPzRN4qucmIiJCNG7cWG1bhYWFwsLCQhw4cECV39nZWRQWFpa7n8uWLRPe3t4l2rOysoSlpaVISEgQQggRHx8vLC0t1b5PNO256devn2jZsmW5Ocpy/fp1ceXKlTJf169fL3PZLVu2CDMzsxLtvXr1ElOmTNFo+zt27BDGxsbi5s2bZc7z6aefihdeeKHUHpjly5cLT0/PUpfTVc+NXh+/sHTpUkycOBGTJk0CAISHh+PAgQNYs2YNwsLCSsy/du1aNGjQQDUYztfXF7Gxsfjiiy/UzukR1VYymQyWZjXjqSrdu3fHmjVrkJeXhw0bNuDy5cuYPn262jxr167Fhg0bkJycjPz8fBQVFZUYrNmsWTO127S7urri3LlzAICLFy/CxMQE/v7+qulNmjRR+6v44sWLcHd3h7u7u6qtadOmsLe3x8WLF9GuXTsAjwdU2tjYqOZxdnaGsbGx2pgMZ2dntV4hAJgzZw7Gjx+veu/g4KDa7tMDZgGgc+fOJXolns4OPP7L+/Tp06qeGgBQKBQoKChAXl4ehg0bhvDwcHh7e6Nv377o378/Bg4cCBOTsn8uhBCS7gx79epVvP/++zh58iQyMjJUPTYpKSlo3rw5xo8fj169eqFx48bo27cvXnnlFVWPgZSc5YmLi8Pff/+t9hkBQEFBAa5evap636JFC5iZmZW7rvz8fJibm5do37p1K7y9vdGqVSsAQOvWreHt7Y3t27eX+CyfR+oxBwAPDw9Jy5VHmzwRERHo168f3Nzcypxn48aNGD16dKnH0cLCotIHqevttFRRURHi4uLUusYAoHfv3mV2jZ04caLE/H369EFsbGyZA+0KCwuRnZ2t9iIi/bOyskKjRo3QsmVLrFixAoWFhVi0aJFq+vfff4/Zs2djwoQJOHjwIBISEhAUFFRiEOjTp7KAxwXeky9Z8f+DnMv7pV3WL/Vn20vbTnnbfsLBwQGNGjVSvZ4urJ7dbmlZrKys1N4rlUosWrQICQkJqte5c+dw5coVmJubw93dHZcuXcKqVatgYWGBqVOn4qWXXip3MLKPj4+kAZ4DBw7EvXv3sH79evz555/4888/AfwzULdt27ZISkrChx9+iPz8fAwfPhyvv/46AEjKWR6lUgk/Pz+14/LkyqpRo0ap5nv2eJbGwcGh1Cv3Nm7ciAsXLsDExET1unDhgtqpKVtbW2RlZZVYNjMzEwBUp/h8fHxw9epVSftbkdNSLi4uKCoqKrF/d+7cgbOz83O3nZycjN9++03VKVGamJgYXLp0qcx57t+/D0dHx+duqyL0VtxkZGRAoVCUOJjOzs5IT08vdZn09PRS5y8uLkZGRkapy4SFhcHOzk71evqvM12TmxhBbiL9QV9EtVloaCi++OIL3Lp1C8DjX5ABAQGYOnUq2rRpg0aNGqn9Ba4JX19fFBcXIzY2VtV26dIl1RcN8LiXJiUlBampqaq2xMREZGVllTrmQld8fX1x9OhRtbbjx48/d5tt27bFpUuX1AqmJ68nvUgWFhYYNGgQVqxYgcOHD+PEiROq3iwzM7MSV8WMGjUKly9fxo8//lhie0KIUr+s7927h4sXL+K9997Dyy+/DF9f31ILAltbWwQGBmL9+vXYsWMHdu3apbrMubyc2mrbti2uXLkCJyenEselrDFDZWnTpg3S09PV9ufcuXOIjY3F4cOH1Yqn6OhonD59GufPnwfwuGfwxo0bJb7HTp8+DSMjIzRq1AjA42P+8OFDrF69utQMT/+MPmvfvn0lirinX/v27StzWT8/P5iamqqudgIeX+V0/vx5BAQEPPfYbNq0CU5OThgwYECZ80RERMDPz0/Vw/Ws8+fPo02bNs/dVkXovf9ak79cnjd/ae1PzJ8/HyEhIar32dnZlVLgONmY49JHlXO/BaLaoFu3bmjWrBmWLFmClStXolGjRvj2229x4MABeHl5YfPmzTh9+jS8vLw0XueT0yGTJ0/G119/DRMTE8yaNQsWFhaqeXr27ImWLVti9OjRCA8PR3FxMaZOnYquXbuWOCWkS3PmzMHw4cPRtm1bvPzyy/j555+xe/du/Pbbb+Uu98EHH+CVV16Bu7s7hg0bBiMjI5w9exbnzp3DRx99hMjISCgUCnTo0AGWlpbYvHkzLCwsVKcyPD09ER0djREjRkAul8PBwQHDhw/Hnj17MHLkSLz//vvo1asXHB0dce7cOSxbtgzTp08vcfO+F154AXXr1sXXX38NV1dXpKSkYN68eWrzLFu2DK6urmjdujWMjIzwww8/wMXFBfb29s/Nqa3Ro0fj888/x+DBg7F48WLUr18fKSkp2L17N+bMmaPV/YXatGkDR0dHHDt2DK+88gqAx1/Y7du3x0svvVRi/k6dOiEiIgLLli1D79694evrixEjRuDjjz+Gm5sbzp49i3fffRfBwcGq02YdOnTA3Llz8c477+DmzZsYOnQo3Nzc8Pfff2Pt2rX417/+hZkzZ5aaryKnpezs7DBx4kS88847qFu3LurUqYN3330XLVq0QM+ePVXzvfzyyxg6dCjefvttVZtSqcSmTZswbty4Mk8fZmdn44cffsCXX35ZZoaYmBh8+OGHkvdBE3rruXFwcICxsXGJ6ra8rjEXF5dS5zcxMUHdunVLXUYul8PW1lbtRUTVU0hICNavX4/U1FQEBwfj1VdfRWBgIDp06IB79+5Jug/Opk2b4O7ujq5du+LVV1/FlClTVPdJAf65y+0LL7yAl156CT179oS3tzd27Nihy10rYciQIVi+fDk+//xzNGvWDOvWrcOmTZvQrVu3cpfr06cP/vvf/yIqKgrt2rVDx44dsXTpUtUXnr29PdavX4/OnTujZcuW+P333/Hzzz+rfkcuXrwY169fR8OGDVWnBmQyGbZu3YqlS5diz5496Nq1K1q2bImFCxdi8ODB6NOnT4kcRkZG2L59O+Li4tC8eXPMnj1bdbO2J6ytrfHpp5/C398f7dq1w/Xr17Fv3z4YGRk9N6e2LC0tER0djQYNGuDVV1+Fr68vJkyYgPz8fK1/7xsbG2PChAnYsmULgH+uDiprbOdrr72G7777DkVFRTAxMcHBgwfh7e2N0aNHo1mzZpg3bx4mTZqEpUuXqi336aefYuvWrfjzzz/Rp08fNGvWDCEhIWjZsiXGjRsn6ThoYtmyZRgyZAiGDx+Ozp07w9LSEj///LPa2LWrV6+WOCPy22+/ISUlBRMmTChz3du3b4cQAiNHjix1+okTJ5CVlaU6PVlZZEJoeectHerQoQP8/PzUuuWaNm2KwYMHlzqg+N///jd+/vlnJCYmqtreeustJCQk4MSJExptMzs7G3Z2dsjKymKhQzVaQUEBkpKSVLdSICLduX37Npo1a4a4uLhKGcBbWw0bNgxt2rTBf/7zn1Knl/d7TZvvb73e5yYkJAQbNmzAxo0bcfHiRcyePRspKSkIDg4G8PiU0tixY1XzBwcHIzk5GSEhIbh48SI2btyIiIgIvPvuu/raBSIiMkDOzs6IiIhASkqKvqMYjMLCQrRq1QqzZ8+u9G3pdcxNYGAg7t27p7qhVPPmzbFv3z5VlZyWlqb2g+Xl5YV9+/Zh9uzZWLVqFdzc3LBixQpeBk5ERDpX2k0NSTq5XI733nuvSral19NS+sDTUmQoeFqKiAyNQZyWIqKKq2V/nxCRAdPV7zMWN0Q11JMbyFX2nT6JiKrKkxtAPn3llhR6v88NEUljbGwMe3t71e3+LS0tJd/OnYhI35RKJe7evQtLS0vJj+F4gsUNUQ3m4uICACWeZ0REVBMZGRmhQYMGFf5DjcUNUQ0mk8ng6uoKJycnyc/kISKqLszMzNQeRisVixsiA2BsbFzhc9RERIaCA4qJiIjIoLC4ISIiIoPC4oaIiIgMSq0bc/PkBkHZ2dl6TkJERESaevK9rcmN/mpdcZOTkwMAcHd313MSIiIi0lZOTg7s7OzKnafWPVtKqVTi1q1bsLGx0fkNz7Kzs+Hu7o7U1FQ+t6oS8ThXDR7nqsHjXHV4rKtGZR1nIQRycnLg5ub23MvFa13PjZGREerXr1+p27C1teX/OFWAx7lq8DhXDR7nqsNjXTUq4zg/r8fmCQ4oJiIiIoPC4oaIiIgMCosbHZLL5QgNDYVcLtd3FIPG41w1eJyrBo9z1eGxrhrV4TjXugHFREREZNjYc0NEREQGhcUNERERGRQWN0RERGRQWNwQERGRQWFxo6XVq1fDy8sL5ubm8PPzQ0xMTLnzHzlyBH5+fjA3N4e3tzfWrl1bRUlrNm2O8+7du9GrVy84OjrC1tYWnTp1woEDB6owbc2l7c/zE8eOHYOJiQlat25duQENhLbHubCwEAsWLICHhwfkcjkaNmyIjRs3VlHamkvb47xlyxa0atUKlpaWcHV1RVBQEO7du1dFaWum6OhoDBw4EG5ubpDJZNi7d+9zl9HL96AgjW3fvl2YmpqK9evXi8TERDFz5kxhZWUlkpOTS53/2rVrwtLSUsycOVMkJiaK9evXC1NTU7Fz584qTl6zaHucZ86cKT799FNx6tQpcfnyZTF//nxhamoqzpw5U8XJaxZtj/MTmZmZwtvbW/Tu3Vu0atWqasLWYFKO86BBg0SHDh1EVFSUSEpKEn/++ac4duxYFaauebQ9zjExMcLIyEgsX75cXLt2TcTExIhmzZqJIUOGVHHymmXfvn1iwYIFYteuXQKA2LNnT7nz6+t7kMWNFtq3by+Cg4PV2po0aSLmzZtX6vxz584VTZo0UWt78803RceOHSstoyHQ9jiXpmnTpmLRokW6jmZQpB7nwMBA8d5774nQ0FAWNxrQ9jjv379f2NnZiXv37lVFPIOh7XH+/PPPhbe3t1rbihUrRP369Ssto6HRpLjR1/cgT0tpqKioCHFxcejdu7dae+/evXH8+PFSlzlx4kSJ+fv06YPY2Fg8evSo0rLWZFKO87OUSiVycnJQp06dyohoEKQe502bNuHq1asIDQ2t7IgGQcpx/umnn+Dv74/PPvsM9erVg4+PD959913k5+dXReQaScpxDggIwI0bN7Bv3z4IIXD79m3s3LkTAwYMqIrItYa+vgdr3YMzpcrIyIBCoYCzs7Nau7OzM9LT00tdJj09vdT5i4uLkZGRAVdX10rLW1NJOc7P+vLLL5Gbm4vhw4dXRkSDIOU4X7lyBfPmzUNMTAxMTPirQxNSjvO1a9dw9OhRmJubY8+ePcjIyMDUqVNx//59jrspg5TjHBAQgC1btiAwMBAFBQUoLi7GoEGD8NVXX1VF5FpDX9+D7LnRkkwmU3svhCjR9rz5S2snddoe5ye2bduGhQsXYseOHXBycqqseAZD0+OsUCgwatQoLFq0CD4+PlUVz2Bo8/OsVCohk8mwZcsWtG/fHv3798fSpUsRGRnJ3pvn0OY4JyYmYsaMGfjggw8QFxeHX3/9FUlJSQgODq6KqLWKPr4H+eeXhhwcHGBsbFzir4A7d+6UqEqfcHFxKXV+ExMT1K1bt9Ky1mRSjvMTO3bswMSJE/HDDz+gZ8+elRmzxtP2OOfk5CA2Nhbx8fF4++23ATz+EhZCwMTEBAcPHkSPHj2qJHtNIuXn2dXVFfXq1YOdnZ2qzdfXF0II3LhxAy+++GKlZq6JpBznsLAwdO7cGXPmzAEAtGzZElZWVujSpQs++ugj9qzriL6+B9lzoyEzMzP4+fkhKipKrT0qKgoBAQGlLtOpU6cS8x88eBD+/v4wNTWttKw1mZTjDDzusRk/fjy2bt3Kc+Ya0PY429ra4ty5c0hISFC9goOD0bhxYyQkJKBDhw5VFb1GkfLz3LlzZ9y6dQsPHz5UtV2+fBlGRkaoX79+peatqaQc57y8PBgZqX8FGhsbA/inZ4EqTm/fg5U6XNnAPLnUMCIiQiQmJopZs2YJKysrcf36dSGEEPPmzRNvvPGGav4nl8DNnj1bJCYmioiICF4KrgFtj/PWrVuFiYmJWLVqlUhLS1O9MjMz9bULNYK2x/lZvFpKM9oe55ycHFG/fn3x+uuviwsXLogjR46IF198UUyaNElfu1AjaHucN23aJExMTMTq1avF1atXxdGjR4W/v79o3769vnahRsjJyRHx8fEiPj5eABBLly4V8fHxqkvuq8v3IIsbLa1atUp4eHgIMzMz0bZtW3HkyBHVtHHjxomuXbuqzX/48GHRpk0bYWZmJjw9PcWaNWuqOHHNpM1x7tq1qwBQ4jVu3LiqD17DaPvz/DQWN5rT9jhfvHhR9OzZU1hYWIj69euLkJAQkZeXV8Wpax5tj/OKFStE06ZNhYWFhXB1dRWjR48WN27cqOLUNcuhQ4fK/X1bXb4HZUKw/42IiIgMB8fcEBERkUFhcUNEREQGhcUNERERGRQWN0RERGRQWNwQERGRQWFxQ0RERAaFxQ0REREZFBY3REREZFBY3BCVIjIyEvb29vqOIZmnpyfCw8PLnWfhwoVo3bp1leSpbv744w80adIESqWySrZXXT4PKduQyWTYu3dvhbY7fvx4DBkypELrKE27du2we/duna+Xaj4WN2Swxo8fD5lMVuL1999/6zsaIiMj1TK5urpi+PDhSEpK0sn6T58+jSlTpqjel/YF9e677+L333/XyfbK8ux+Ojs7Y+DAgbhw4YLW69FlsTl37lwsWLBA9eDE2vJ51CTR0dEYOHAg3Nzcyiyw3n//fcybN6/KilSqOVjckEHr27cv0tLS1F5eXl76jgXg8ZO209LScOvWLWzduhUJCQkYNGgQFApFhdft6OgIS0vLcuextrZG3bp1K7yt53l6P3/55Rfk5uZiwIABKCoqqvRtl+b48eO4cuUKhg0bVmZOQ/48aorc3Fy0atUKK1euLHOeAQMGICsrCwcOHKjCZFQTsLghgyaXy+Hi4qL2MjY2xtKlS9GiRQtYWVnB3d0dU6dOxcOHD8tcz19//YXu3bvDxsYGtra28PPzQ2xsrGr68ePH8dJLL8HCwgLu7u6YMWMGcnNzy80mk8ng4uICV1dXdO/eHaGhoTh//ryqZ2nNmjVo2LAhzMzM0LhxY2zevFlt+YULF6JBgwaQy+Vwc3PDjBkzVNOePg3i6ekJABg6dChkMpnq/dOnKA4cOABzc3NkZmaqbWPGjBno2rWrzvbT398fs2fPRnJyMi5duqSap7zP4/DhwwgKCkJWVpaqZ2XhwoUAgKKiIsydOxf16tWDlZUVOnTogMOHD5ebZ/v27ejduzfMzc3LzGnIn8fTTp8+jV69esHBwQF2dnbo2rUrzpw5U2K+tLQ09OvXDxYWFvDy8sIPP/ygNv3mzZsIDAzECy+8gLp162Lw4MG4fv26xjlK069fP3z00Ud49dVXy5zH2NgY/fv3x7Zt2yq0LTI8LG6oVjIyMsKKFStw/vx5fPPNN/jjjz8wd+7cMucfPXo06tevj9OnTyMuLg7z5s2DqakpAODcuXPo06cPXn31VZw9exY7duzA0aNH8fbbb2uVycLCAgDw6NEj7NmzBzNnzsQ777yD8+fP480330RQUBAOHToEANi5cyeWLVuGdevW4cqVK9i7dy9atGhR6npPnz4NANi0aRPS0tJU75/Ws2dP2NvbY9euXao2hUKB77//HqNHj9bZfmZmZmLr1q0AoDp+QPmfR0BAAMLDw1U9K2lpaXj33XcBAEFBQTh27Bi2b9+Os2fPYtiwYejbty+uXLlSZobo6Gj4+/s/N2tt+DxycnIwbtw4xMTE4OTJk3jxxRfRv39/5OTkqM33/vvv47XXXsNff/2FMWPGYOTIkbh48SIAIC8vD927d4e1tTWio6Nx9OhRWFtbo2/fvmX2zj05DagL7du3R0xMjE7WRQak0p87TqQn48aNE8bGxsLKykr1ev3110ud9/vvvxd169ZVvd+0aZOws7NTvbexsRGRkZGlLvvGG2+IKVOmqLXFxMQIIyMjkZ+fX+oyz64/NTVVdOzYUdSvX18UFhaKgIAAMXnyZLVlhg0bJvr37y+EEOLLL78UPj4+oqioqNT1e3h4iGXLlqneAxB79uxRmyc0NFS0atVK9X7GjBmiR48eqvcHDhwQZmZm4v79+xXaTwDCyspKWFpaCgACgBg0aFCp8z/xvM9DCCH+/vtvIZPJxM2bN9XaX375ZTF//vwy121nZye+/fbbEjlrw+fx7DaeVVxcLGxsbMTPP/+sljU4OFhtvg4dOoi33npLCCFERESEaNy4sVAqlarphYWFwsLCQhw4cEAI8fj/xcGDB6um7969WzRu3LjMHM8q7Xg98eOPPwojIyOhUCg0Xh8ZPvbckEHr3r07EhISVK8VK1YAAA4dOoRevXqhXr16sLGxwdixY3Hv3r0yu/RDQkIwadIk9OzZE5988gmuXr2qmhYXF4fIyEhYW1urXn369IFSqSx3QGpWVhasra1Vp2KKioqwe/dumJmZ4eLFi+jcubPa/J07d1b9tTxs2DDk5+fD29sbkydPxp49e1BcXFyhYzV69GgcPnwYt27dAgBs2bIF/fv3xwsvvFCh/bSxsUFCQgLi4uKwdu1aNGzYEGvXrlWbR9vPAwDOnDkDIQR8fHzUMh05ckTt83lWfn5+iVNSQO35PJ52584dBAcHw8fHB3Z2drCzs8PDhw+RkpKiNl+nTp1KvH+y73Fxcfj7779hY2OjylGnTh0UFBSU+TkMHToU//vf/7Q6HmWxsLCAUqlEYWGhTtZHhsFE3wGIKpOVlRUaNWqk1pacnIz+/fsjODgYH374IerUqYOjR49i4sSJePToUanrWbhwIUaNGoVffvkF+/fvR2hoKLZv346hQ4dCqVTizTffVBtj8USDBg3KzGZjY4MzZ87AyMgIzs7OsLKyUpv+bLe9EELV5u7ujkuXLiEqKgq//fYbpk6dis8//xxHjhxRO92jjfbt26Nhw4bYvn073nrrLezZswebNm1STZe6n0ZGRqrPoEmTJkhPT0dgYCCio6MBSPs8nuQxNjZGXFwcjI2N1aZZW1uXuZyDgwMePHhQor22fB5PGz9+PO7evYvw8HB4eHhALpejU6dOGg32frLvSqUSfn5+2LJlS4l5HB0dNcpREffv34elpaXqNCIRwOKGaqHY2FgUFxfjyy+/VF0K/P333z93OR8fH/j4+GD27NkYOXIkNm3ahKFDh6Jt27a4cOFCiSLqeZ7+0n+Wr68vjh49irFjx6rajh8/Dl9fX9V7CwsLDBo0CIMGDcK0adPQpEkTnDt3Dm3bti2xPlNTU42u+hk1ahS2bNmC+vXrw8jICAMGDFBNk7qfz5o9ezaWLl2KPXv2YOjQoRp9HmZmZiXyt2nTBgqFAnfu3EGXLl003n6bNm2QmJhYor02fh4xMTFYvXo1+vfvDwBITU1FRkZGiflOnjyptu8nT55EmzZtVDl27NgBJycn2NraSs4i1fnz50s9xlS78bQU1ToNGzZEcXExvvrqK1y7dg2bN28ucZrkafn5+Xj77bdx+PBhJCcn49ixYzh9+rTqi+3f//43Tpw4gWnTpiEhIQFXrlzBTz/9hOnTp0vOOGfOHERGRmLt2rW4cuUKli5dit27d6sG0kZGRiIiIgLnz59X7YOFhQU8PDxKXZ+npyd+//13pKenl9pr8cTo0aNx5swZfPzxx3j99dfVTt/oaj9tbW0xadIkhIaGQgih0efh6emJhw8f4vfff0dGRgby8vLg4+OD0aNHY+zYsdi9ezeSkpJw+vRpfPrpp9i3b1+Z2+/Tpw+OHj2qVWZD/TwaNWqEzZs34+LFi/jzzz8xevToUntAfvjhB2zcuBGXL19GaGgoTp06pRq4PHr0aDg4OGDw4MGIiYlBUlISjhw5gpkzZ+LGjRulbnfPnj1o0qRJudkePnyoOp0MAElJSUhISChxyiwmJga9e/fWeJ+pltDvkB+iyvPsIManLV26VLi6ugoLCwvRp08f8e233woA4sGDB0II9QGmhYWFYsSIEcLd3V2YmZkJNzc38fbbb6sN2jx16pTo1auXsLa2FlZWVqJly5bi448/LjNbaQNkn7V69Wrh7e0tTE1NhY+Pj9og2D179ogOHToIW1tbYWVlJTp27Ch+++031fRnB7D+9NNPolGjRsLExER4eHgIIcoeXNquXTsBQPzxxx8lpulqP5OTk4WJiYnYsWOHEOL5n4cQQgQHB4u6desKACI0NFQIIURRUZH44IMPhKenpzA1NRUuLi5i6NCh4uzZs2Vmun//vrCwsBD/+9//npvzaYbweTy7jTNnzgh/f38hl8vFiy++KH744YdSBz+vWrVK9OrVS8jlcuHh4SG2bdumtt60tDQxduxY4eDgIORyufD29haTJ08WWVlZQoiS/y8+GWhenkOHDqkGoD/9GjdunGqeGzduCFNTU5Gamlruuqj2kQkhhH7KKiIi/Zg7dy6ysrKwbt06fUehCpgzZw6ysrLw9ddf6zsKVTM8LUVEtc6CBQvg4eGhk7sPk/44OTnhww8/1HcMqobYc0NEREQGhT03REREZFBY3BAREZFBYXFDREREBoXFDRERERkUFjdERERkUFjcEBERkUFhcUNEREQGhcUNERERGRQWN0RERGRQ/g+tGPDBfh07BAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics.plot_roc_curve(forest, X_test, y_test)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Q7. Based on the random forest model, which of the following is the most important feature in predicting the target variable?\n",
    "\n",
    "\n",
    "To answer this questio, you need to get the importance of each variable which have been saved in **feature_importance_**.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "importances = forest.feature_importances_\n",
    "\n",
    "df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})\n",
    "\n",
    "df = df.sort_values('importance')\n",
    "\n",
    "print(df)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                    feature  importance\n",
      "21              purpose_A48    0.001737\n",
      "15             purpose_A410    0.002645\n",
      "18              purpose_A44    0.003130\n",
      "19              purpose_A45    0.005668\n",
      "25      savings_balance_A64    0.007269\n",
      "31        other_credit_A142    0.009528\n",
      "24      savings_balance_A63    0.009597\n",
      "34             housing_A153    0.009988\n",
      "20              purpose_A46    0.011386\n",
      "23      savings_balance_A62    0.011644\n",
      "22              purpose_A49    0.012295\n",
      "12       credit_history_A33    0.012547\n",
      "8      checking_balance_A13    0.012949\n",
      "29  employment_duration_A74    0.014069\n",
      "37                 job_A174    0.014574\n",
      "14              purpose_A41    0.015595\n",
      "35                 job_A172    0.015615\n",
      "6                dependents    0.016054\n",
      "30  employment_duration_A75    0.016573\n",
      "16              purpose_A42    0.017283\n",
      "11       credit_history_A32    0.017431\n",
      "28  employment_duration_A73    0.018203\n",
      "10       credit_history_A31    0.018878\n",
      "36                 job_A173    0.018904\n",
      "27  employment_duration_A72    0.019264\n",
      "26      savings_balance_A65    0.019882\n",
      "17              purpose_A43    0.020067\n",
      "7      checking_balance_A12    0.021254\n",
      "33             housing_A152    0.021878\n",
      "32        other_credit_A143    0.022411\n",
      "38               phone_A192    0.023109\n",
      "5      existing_loans_count    0.023235\n",
      "13       credit_history_A34    0.024321\n",
      "3        years_at_residence    0.042047\n",
      "2         percent_of_income    0.044770\n",
      "9      checking_balance_A14    0.068948\n",
      "4                       age    0.105429\n",
      "0      months_loan_duration    0.109518\n",
      "1                    amount    0.140305\n"
     ]
    }
   ],
   "source": [
    "importances = forest.feature_importances_\n",
    "\n",
    "df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})\n",
    "\n",
    "df = df.sort_values('importance')\n",
    "\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SF_tYkbKeU1u"
   },
   "source": [
    "# Step 5: Grid Search\n",
    "\n",
    "In this step we will improve the model’s performance by tuning its parameters.\n",
    "\n",
    "#### Q8. Apply a hyperparameter tuning model using GridSearch algorithm on the training set and evaluate the model using the test set. What is the model performance??\n",
    "\n",
    "To answer this question, use **GridSearchCV()** (set **random_state =0**)to tune the parameters in the decision tree model.The parameters we will tuned include: **criterion (gini or entropy)**, **max_leaf_nodes (ranges from 2 to 50)**, **max_depth(ranges from 3 to 15)**. In addition we perform cross validation as well.\n",
    "\n",
    "1. first we define a combination of parameters and apply them to the decision tree model.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "params = {'criterion':['gini','entropy'],'max_leaf_nodes': list(range(2, 50)), 'max_depth': np.arange(3, 15)}\n",
    "\n",
    "tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, cv=10)\n",
    "\n",
    "tree_grid.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 381
    },
    "executionInfo": {
     "elapsed": 40020,
     "status": "ok",
     "timestamp": 1601701056591,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "2bE44VTmfCEz",
    "outputId": "a309580d-3cbc-4b92-ce08-797d8a6a5d9d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=10, estimator=DecisionTreeClassifier(random_state=0),\n",
       "             param_grid={'criterion': ['gini', 'entropy'],\n",
       "                         'max_depth': array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]),\n",
       "                         'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,\n",
       "                                            13, 14, 15, 16, 17, 18, 19, 20, 21,\n",
       "                                            22, 23, 24, 25, 26, 27, 28, 29, 30,\n",
       "                                            31, ...]})"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "params = {'criterion':['gini','entropy'],'max_leaf_nodes': list(range(2, 50)), 'max_depth': np.arange(3, 15)}\n",
    "\n",
    "tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, cv=10)\n",
    "\n",
    "tree_grid.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kXmK3-ZMhnBC"
   },
   "source": [
    "#### Q9. What is the value of max_depth of the best model returned by hyperparameter tuning?\n",
    "\n",
    "To answer this question, we can get the parameters of the best model using **best_estimator_** attribute.\n",
    "\n",
    "##### code for reference\n",
    "\n",
    "tree_grid.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 121
    },
    "executionInfo": {
     "elapsed": 383,
     "status": "ok",
     "timestamp": 1601701076009,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "KsVUefrKhncO",
    "outputId": "b5fad147-b986-4ba2-f0e7-47450a0d92dc"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(max_depth=10, max_leaf_nodes=17, random_state=0)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree_grid.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3. Then we apply this model on the test set and get the accuracy.\n",
    "\n",
    "##### code for reference\n",
    "y_pred_grid = tree_grid.predict(X_test)\n",
    "\n",
    "print(\"Grid-search Model Accuracy on test set: {:.3f}\".format(accuracy_score(y_test, y_pred_grid)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 559
    },
    "executionInfo": {
     "elapsed": 899,
     "status": "ok",
     "timestamp": 1601700796227,
     "user": {
      "displayName": "Xiqing Sha",
      "photoUrl": "",
      "userId": "00149820978001691006"
     },
     "user_tz": 420
    },
    "id": "xtUDodfTh5Cm",
    "outputId": "b5aed775-b917-49cb-9963-2d65cc5a4c11"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Grid-search Model Accuracy on test set: 0.740\n"
     ]
    }
   ],
   "source": [
    "y_pred_grid = tree_grid.predict(X_test)\n",
    "\n",
    "print(\"Grid-search Model Accuracy on test set: {:.3f}\".format(accuracy_score(y_test, y_pred_grid)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "4. Lastly we visualize ROC curve and estimate the AUC value."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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cSOuX7+OkJAQnDlzRtFxecyYMXB0dMTQoUMRHh6O2NhYHD16FNOnT0diYmKF5929ezeaNWv21NhycnIUzckAEBsbi6ioqHJNZuHh4ejbt6/K10x1hG67/BBVnyc7MT5u6dKlws3NTVhYWIh+/fqJH374QQAQ6enpQgjlDqaFhYVi1KhRokGDBkIqlQp3d3fx7rvvKnXaPHPmjOjTp4+wtrYWVlZWonXr1uLTTz+tNLaKOsg+afXq1cLHx0eYmpoKPz8/pU6wu3fvFp06dRK2trbCyspKdO7cWfz111+Kz5/swLp3717h6+srTExMhJeXlxCi8s6lHTp0EADE33//Xe4zbV1nXFycMDExEdu3bxdCPPv7EEKIoKAg4eDgIACIkJAQIYQQRUVFYsGCBcLb21uYmpoKV1dXMWzYMHHx4sVKY3r48KGwsLAQV69efWacjzOE7+PJc5w/f14EBgYKMzMz0aRJE/HLL79U2Pl51apVok+fPsLMzEx4eXmJrVu3Kh03KSlJjB07Vjg6OgozMzPh4+Mj3nzzTZGZmSmEKP//YllH86c5fPiwogP6469x48YptklMTBSmpqYiISHhqceiukcihBC6SauIiHRjzpw5yMzMxLfffqvrUKgKZs+ejczMTHz33Xe6DoVqGTZLEVGdM3/+fHh5eWll9mHSHWdnZ3z88ce6DoNqIdbcEBERkUFhzQ0REREZFCY3REREZFCY3BAREZFBYXJDREREBoXJDRERERkUJjdERERkUJjcEBERkUFhckNEREQGhckNERERGZT/Bw8ytaBffynEAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics.plot_roc_curve(tree_grid, X_test, y_test)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Q10. Which model performs the best (using AUC as the evaluation metric)?\n",
    "\n",
    "To answer this question, you need to compare the models created and select the one with highest AUC."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyO/KY3gsEBKmU0EWBjMggrA",
   "name": "Model Evaluation and Improvement.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
