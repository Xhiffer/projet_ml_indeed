{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# A IMPORTER\n",
    "\n",
    "import pandas as pd\n",
    "import json\n",
    "import unidecode\n",
    "import re\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cleaning the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On charge notre dataset. Puis on créé un autre dataset reprenant les features qui nous intéresse pour appliquer nos processus NLP."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_json('indeed6.json', encoding = 'utf-8', lines=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "job_to_class = data[['titre', 'texte']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ici, nous avons décidé d'utiliser la libraire spaCy. C'est elle qui nous permettra de parseriser nos descriptions et nos titres, retirer les stop words, et tokenizer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy \n",
    "#Chargeons notre vocabulaire spaCy\n",
    "nlp_fr = spacy.load(\"fr_core_news_sm\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Chargeons nos stopwords qu'on mettra dans une liste\n",
    "from spacy.lang.en.stop_words import STOP_WORDS\n",
    "from spacy.lang.fr.stop_words import STOP_WORDS\n",
    "french_stopwords = list(spacy.lang.fr.stop_words.STOP_WORDS)\n",
    "english_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)\n",
    "stop_words = french_stopwords + english_stopwords\n",
    "\n",
    "#J'importe la ponctuation qui nous servira au traitement du texte\n",
    "import string\n",
    "punctuations = string.punctuation\n",
    "\n",
    "#Ainsi que notre Parser qui comprend la structure grammatical de notre texte\n",
    "from spacy.lang.fr import French\n",
    "parser = French()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dans un premier temps, on avait opté pour utiliser strictement spaCy pour le traitement global. Toutefois, il restait imparfait avec beaucoup de \"déchet\" dans nos tokens.\n",
    "On choisit donc de subdiviser à nouveau notre traitement. \n",
    "Tout d'abord on retire les caractères indésirables. Puis, dans la mesure où on a déjà créé des nouvelles features grâce au regex, on supprime ces éléments de nos données afin d'éviter tout risque de colinéarité dans l'entraînement de nos modèles de machine learning.\n",
    "Enfin on tokenize nos textes et on nettoie à nouveau les caractères indésirables qui ont pu se glisser dedans."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [],
   "source": [
    "from string import digits\n",
    "\n",
    "#Ces caractères indésirables ne nous fournissent pas d'information pertinente. On les retire\n",
    "def clean_and_lower(string):\n",
    "    forbid_car= [\":\", \";\", \",\", \"&\", \"(\", \")\",\n",
    "                '\"', \"!\", \"?\", \"*\", \"-\", \"\\n\", \n",
    "                \"...\", \"/\",\"'\"]\n",
    "    for car in forbid_car:\n",
    "            cleaned_string = string.replace(car, ' ')\n",
    "    return cleaned_string\n",
    "\n",
    "#On retire les termes déjà étudié lors de nos regex, et qui sont déjà des features\n",
    "def terme_a_retirer(string):\n",
    "        string = re.sub(pattern = '(jeune[es().]* diplôme[es().]*|junior[s]*|novice[s]*| débutant[es().]*)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(expérimenté[es().]*|senior[s]*|confirmé[es().]*)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(licence|bac[ +]*[23/]+|iut|dut|bts| bsc|master|bac[ +]*5|ingénieur[.()es]*|grande[s]* école[s]*| msc|doctorat[s]*|docteur[.()es]*|ph[.]*d|thèse[s]*|bac)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(minimum [1-9]*[/aou-]*\\d+ an[nées]* d\\'expérience|[1-9]*[/aou-]*\\d+ an[nées]* d\\'expérience|[1-9]*[/aou-]*\\d+ an[nées]* minimum d\\'expérience|expérience minimum de *[1-9]*[/aou-]*\\d+ an[nees]|expérience|d\\'expérience)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(nantes|bordeaux|paris|île-de-france|lyon|toulouse)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '[\\d+ ]*[ \\-k€]*[\\d+ ]*\\d+,?\\d+[ ]*[k€$]+', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(cdi|CDI|Cdi|cdd|CDD|Cdd|stage[s]*|stagiaire[s]*|intern[s]*|internship[s]*|freelance|Freelance|FREELANCE||indépendant[s]*|par an|/an|temps plein|3dexperience)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(h[/ \\-]*f|f[/ \\-]*h)', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '(salaire|[\\d+]+[ ]*an[nées]*|[\\d+]+[ ]*jour[s]|/mois|/jour[s]*|/semaine[s]*|[\\d+]+ mois|[\\d+]+ semaine[s]*|mois|semaine[s]*|jour[s]*])', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '[\\d+]+[]*[èe]*[mes]*', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = r'\\d+', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        string = re.sub(pattern = '[\\w\\.-]+@[\\w\\.-]+', repl = '', string = string, flags=re.IGNORECASE)\n",
    "        return string\n",
    "\n",
    "#On tokenize notre texte, on retire les pronoms, stop words, et on réduit chaque terme à sa racine\n",
    "def tokenize(string):\n",
    "    parsed_string = parser(string)\n",
    "    tokenized_string = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in parsed_string]\n",
    "    cleaned_tokenized_string = [word for word in tokenized_string if word not in punctuations and word not in stop_words]\n",
    "    return cleaned_tokenized_string\n",
    "\n",
    "#On retire la ponctuaction indésirable qui peut rester dans nos tokens\n",
    "def strip_final_punctuation(list_of_tokens):\n",
    "    forbid_car= [\":\", \";\", \",\", \"&\", \"(\", \")\",\n",
    "                '\"', \"!\", \"?\", \"*\", \"-\", \n",
    "                \"...\", \"/\",\"'\", \"\\\\\", \"·\", \".\", \"∙\", \"•\"\n",
    "                \"–\"]\n",
    "    \n",
    "    for i in range(len(list_of_tokens)):\n",
    "        for car in forbid_car:\n",
    "            list_of_tokens[i] = list_of_tokens[i].replace(car, '')\n",
    "    return list_of_tokens\n",
    "    \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On définit ensuite une fonction finale de nettoyage qui tournera sur nos descriptions et sur nos titres."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cleaning(string):\n",
    "    cleaned_string = clean_and_lower(string)\n",
    "    cleaned_string = terme_a_retirer(cleaned_string)\n",
    "    tokenized_string = tokenize(cleaned_string)\n",
    "    tokenized_string = strip_final_punctuation(tokenized_string)\n",
    "    return tokenized_string\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Traitement des descriptions avec SpaCy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### I. Tokenize, clean, lemmatize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\David\\Miniconda3\\lib\\site-packages\\ipykernel_launcher.py:3: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  This is separate from the ipykernel package so we can avoid doing imports until\n"
     ]
    }
   ],
   "source": [
    "#On tokenize complètement avec spaCy nos descriptions\n",
    "\n",
    "job_to_class['spacy_description'] = [cleaning(job_to_class.loc[i, 'texte']) for i in range(len(job_to_class['texte']))]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### II. Create the TF-IDF term-documents matrix with the processed data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Il nous faut maintenant évaluer le poid relatif de chaque terme dans nos descriptions. On choisit donc de vectoriser en utilisant une matrice Tf-Idf."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "L'initialisation des paramètres de notre vectorisation reste encore une question ouverte. L'idéal aurait été de tester automatiquement plusieurs nombre maximal de features et plusieurs n_gram, les stocker dans un log, et choisir le plus adapté. Faute de temps, on a choisi un grand nombre de features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "metadata": {},
   "outputs": [],
   "source": [
    "#On initialise nos paramètres pour la vectorisation\n",
    "\n",
    "max_features_description = 5000\n",
    "min_n_description = 1\n",
    "max_n_description = 3\n",
    "\n",
    "#On crée une fonction inutile. En effet TfidfVectorizer ne peux pas gérer des documents déjà tokenisés\n",
    "\n",
    "def dummy_fun(doc): \n",
    "    return doc\n",
    "\n",
    "#On vectorise\n",
    "\n",
    "tfidf_vectorizer_description = TfidfVectorizer( analyzer = 'word',\n",
    "                                    tokenizer = dummy_fun,\n",
    "                                    preprocessor = dummy_fun,\n",
    "                                    token_pattern = None,\n",
    "                                    max_df = 1.0, min_df = 1,\n",
    "                                    max_features = max_features_description, sublinear_tf = True,\n",
    "                                    ngram_range=(min_n_description, max_n_description))\n",
    "\n",
    "\n",
    "tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(job_to_class['spacy_description'])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### III. Get the features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On utilise maintenant notre matrice term-document pour extraire les features les plus importantes de nos descriptions. C'est celles-ci qui vont ensuite nous servir pour l'extraction de topic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "metadata": {},
   "outputs": [],
   "source": [
    "features_description = tfidf_vectorizer_description.get_feature_names()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['', ' profil', ' yu', ' yu continuer', '7e', '7e externe', 'ability', 'ability work', 'able', 'aborder', 'accenture', 'accepter', 'access', 'accessible', 'accompagnement', 'accompagnement client', 'accompagnement client dan', 'accompagner', 'accompagner client', 'accompagner client dan', 'accompagner croissance', 'accompagner dan', 'accompagner dan évolution', 'accompagner développement', 'accompagner entreprise', 'accompagner quotidien', 'accompagner transformation', 'accompagner équiper', 'accompagner évolution', 'accomplir', 'accomplish', 'accomplish extraordinary', 'accomplish extraordinary ground', 'accord', 'according', 'account', 'account manager', 'accounting', 'accounts', 'accroître', 'accueil', 'accueillir', 'accès', 'accélération', 'accélérer', 'accélérer cgi', 'accélérer cgi plan', 'achat', 'achats', 'achieve', 'achieve excellence', 'achieve excellence diversity', 'achieving', 'achieving workforce', 'achieving workforce diversity', 'acquis', 'acquisition', 'acquérir', 'act', 'act integrity', 'act integrity foundation', 'acteur', 'acteur majeur', 'acteur mondial', 'acteur secteur', 'actif', 'action', 'actionnaire', 'actionnaire propriétaire', 'actionnaire propriétaire professionnel', 'active', 'active directory', 'actively', 'activement', 'activer', 'activities', 'activity', 'activité', 'activités', 'activus', 'activus activus', 'activus activus group', 'activus group', 'activus group société', 'actualité', 'actuel', 'actuellement', 'ad', 'ad hoc', 'adaptabilité', 'adaptable', 'adaptation', 'adapter', 'adapter besoin', 'additional', 'additional information', 'adepte', 'administrateur', 'administratif', 'administration', 'administration système', 'administrer', 'adn', 'adn excellence', 'adn excellence technologique', 'adobe', 'adonis', 'adopter', 'adoption', 'adresser', 'ads', 'advanced', 'adventure', 'advertising', 'advisory', 'adéquat', 'adéquation', 'aeronautics', 'aeronautics space', 'aeronautics space related', 'affairer', 'affairer milliard', 'affairer monder', 'affairer monder savoir', 'affaires', 'afrique', 'afterworks', 'age', 'age gender', 'age gender disability', 'agence', 'agencer', 'agencer aix', 'agencer aix marseille', 'agencies', 'agency', 'agent', 'agile', 'agile devops', 'agile scrum', 'agiles', 'agilité', 'agilité associer', 'agilité associer diversité', 'agir', 'agricole', 'agréable', 'aider', 'aider client', 'aider décision', 'aimer', 'aimer challenge', 'aimer partager', 'aimer travail', 'aimer travail équiper', 'aimer travailler', 'aimer travailler équiper', 'air', 'airbus', 'airbus airbus', 'airbus airbus committed', 'airbus committed', 'airbus committed achieving', 'airbus european', 'airbus european leader', 'airbus global', 'airbus global leader', 'airbus offers', 'airbus offers comprehensive', 'airbus provides', 'airbus provides efficient', 'airbus storing', 'airbus storing information', 'aircraft', 'aircraft world', 'aircraft world leading', 'airliners', 'airliners airbus', 'airliners airbus european', 'aisance', 'aisance relationnel', 'aise', 'aise dan', 'aix', 'aix marseille', 'aix marseille recruter', 'ajax', 'ajouter', 'akka', 'al', 'al external', 'alerter', 'algorithme', 'algorithms', 'alimentation', 'alimenter', 'allemagne', 'aller', 'allianz', 'allier', 'allow', 'ally', 'alm', 'alteca', 'alternance', 'amazing', 'amazon', 'amazon web', 'amazon web services', 'ambiance', 'ambiance travail', 'ambitieux', 'ambition', 'amener', 'amener intervenir', 'amener travailler', 'amenée', 'amoa', 'amont', 'amélioration', 'amélioration continuer', 'améliorer', 'améliorer performance', 'aménagement', 'analyse', 'analyse besoin', 'analyser', 'analyser besoin', 'analyser donner', 'analyser performance', 'analyser syn', 'analysis', 'analyst', 'analyste', 'analysts', 'analytical', 'analytical skills', 'analytics', 'analytics datascience', 'analytics datascience conseil', 'analytique', 'analyze', 'android', 'anglais', 'anglais courir', 'anglais technique', 'angler', 'angler courir', 'angler professionnel', 'angler technique', 'angler écrire', 'angular', 'angular react', 'angularjs', 'animation', 'animer', 'annoncer', 'annonceur', 'annual', 'annuel', 'année', 'anomalie', 'ansible', 'anticiper', 'apache', 'apecfr', 'apecfr yu', 'apecfr yu continuer', 'api', 'api rest', 'apis', 'app', 'appel', 'appel offrir', 'applicable', 'applicants', 'applicatif', 'applicatifs', 'application', 'application consenting', 'application consenting airbus', 'application dan', 'application existant', 'application futur', 'application futur employment', 'application irrespective', 'application irrespective social', 'application mobile', 'application métier', 'application web', 'applications', 'applicative', 'applicatives', 'appliquer', 'apply', 'apporter', 'apporter expertiser', 'apporter solution', 'apprendre', 'apprentissage', 'apprentissage contrat', 'apprentissage contrat pro', 'approach', 'approcher', 'approcher renforcer', 'approcher renforcer culture', 'approfondir', 'appropriate', 'approprier', 'apprécier', 'apprécier travail', 'apprécier travail équiper', 'appréhender', 'apps', 'apps datavizualisation', 'apps datavizualisation sécurité', 'appui', 'appuyer', 'appuyer proximité', 'appuyer proximité équiper', 'appétence', 'aptitude', 'architect', 'architecte', 'architects', 'architectural', 'architecture', 'architecture technique', 'architecturer', 'architecturer logicielle', 'architecturer technique', 'area', 'areas', 'arriver', 'arrondissement', 'art', 'article', 'articuler', 'articuler autour', 'artificial', 'artificial intelligence', 'artificiel', 'artificielle', 'asap', 'asie', 'aspect', 'aspiration', 'aspnet', 'assessment', 'asset', 'asset management', 'assets', 'assigned', 'assist', 'assistance', 'assistance technique', 'assistant', 'assister', 'associate', 'association', 'associer', 'associer diversité', 'associer diversité client', 'assurance', 'assurance ingeniance', 'assurance ingeniance également', 'assurance offrir', 'assurance offrir aujourd’hui', 'assurances', 'assurer', 'assurer bon', 'assurer développement', 'assurer l', 'assurer maintenance', 'assurer support', 'assurer veiller', 'assurer veiller technologique', 'atelier', 'ational', 'ationale', 'ationales', 'ationaux', 'atos', 'atout', 'atout réussir', 'atr', 'attacher', 'atteindre', 'atteindre objectif', 'atteint', 'attendre', 'attendre plaire', 'attenter', 'attention', 'attitude', 'attractif', 'audelà', 'audience', 'audit', 'audit conseil', 'audit user', 'audit user experience', 'augmenter', 'aujourd’hui', 'aujourd’hui plaire', 'aujourd’hui équiper', 'aujourd’hui équiper perspective', 'auprès', 'auprès client', 'auprès d', 'auprès grand', 'auprès équiper', 'autant', 'auto', 'automate', 'automated', 'automation', 'automatique', 'automatisation', 'automatisation test', 'automatiser', 'automobile', 'autonome', 'autonome rigoureux', 'autonomie', 'autonomie rigueur', 'autonomous', 'autonomy', 'autour', 'autour big', 'autour big data', 'availability', 'available', 'avancement', 'avancer', 'avantager', 'avantages', 'avantgardiste', 'avantgardiste technologie', 'avantgardiste technologie disruptives', 'avantvente', 'avenir', 'aventurer', 'avérer', 'awareness', 'awareness potential', 'awareness potential compliance', 'aws', 'aws azure', 'axa', 'axa yu', 'axa yu continuer', 'axer', 'azure', 'azure devops', 'aéronautique', 'à', 'b', 'babyfoot', 'balancer', 'bancaire', 'banking', 'banque', 'banque assurance', 'banque finance', 'banque finance assurance', 'banque l', 'banque l assurance', 'base', 'base donner', 'based', 'baser', 'baser dan', 'baser donner', 'baser donner relationnel', 'baser donner sql', 'bases', 'bases donner', 'bash', 'basic', 'basis', 'batch', 'bb', 'bc', 'bdd', 'belgique', 'belief', 'belief yu', 'belief yu continuer', 'believe', 'belle', 'benchmark', 'benefits', 'besoin', 'besoin client', 'besoin fonctionnel', 'besoin métier', 'besoin utilisateur', 'best', 'best practices', 'better', 'bi', 'bi big', 'bi big data', 'bienveillance', 'bienveillant', 'bienvenue', 'bienêtre', 'big', 'big data', 'big data analytics', 'big data développement', 'big data intelligence', 'big dater', 'bigdata', 'bilan', 'billion', 'billion employed', 'billion employed workforce', 'blank', 'blockchain', 'blockchain philosophie', 'blockchain philosophie devops', 'blog', 'bnp', 'bnp paribas', 'bo', 'boire', 'bon', 'bon capacité', 'bon capacité d', 'bon communication', 'bon compréhension', 'bon connaissance', 'bon déroulement', 'bon esprit', 'bon fonctionnement', 'bon humeur', 'bon maîtriser', 'bon niveau', 'bon niveau angler', 'bon niveau d', 'bon relationnel', 'bon sentir', 'bon vivre', 'bonjour', 'bonne', 'bonne capacité', 'bonne connaissance', 'bonne pratiquer', 'bonne pratiquer développement', 'bonnes', 'bonnes connaissance', 'bonus', 'boot', 'bootstrap', 'bord', 'bouillir', 'bouillir bouillir', 'boulognebillancourt', 'bouygues', 'bpce', 'brancher', 'brand', 'brands', 'bref', 'bring', 'brings', 'brique', 'broad', 'brut', 'btob', 'bu', 'budget', 'budgétaire', 'bug', 'build', 'building', 'built', 'bureau', 'bureautique', 'business', 'business analyst', 'business developer', 'business development', 'business intelligence', 'business intelligence big', 'business unit', 'businesses', 'bâtiment', 'bénéfice', 'bénéfice issu', 'bénéfice issu croissance', 'bénéficier', 'bénéficier d', 'bénéficier valeur', 'bénéficier valeur créer', 'c', 'c c', 'c net', 'cabinet', 'cabinet conseil', 'cabinet recrutement', 'cabinet recrutement retenue', 'cac', 'cadrage', 'cadre', 'cadrer', 'cadrer d', 'cadrer développement', 'cadrer développement activité', 'cadrer mission', 'cadrer projet', 'cadrer travail', 'café', 'cahier', 'cahier charger', 'calcul', 'campagne', 'campaign', 'campaigns', 'canada', 'canal', 'candidat', 'candidat niveau', 'candidat niveau d', 'candidate', 'candidature', 'candidature solliciter', 'candidature solliciter provenir', 'capabilities', 'capability', 'capable', 'capacity', 'capacité', 'capacité adaptation', 'capacité analyser', 'capacité analyser syn', 'capacité communication', 'capacité d', 'capacité d adaptation', 'capacité d analyser', 'capacité rédactionnel', 'capacité travailler', 'capacité travailler équiper', 'capacité écouter', 'capacités', 'capgemini', 'capital', 'capitalisation', 'capitaliser', 'caractère', 'care', 'career', 'carriere', 'carriere info', 'carriere info yu', 'carrière', 'carrière secteur', 'carrière secteur technologie', 'carrière stimulant', 'carrière stimulant réussite', 'carte', 'cartographie', 'cas', 'caser', 'cassandra', 'cataloguer', 'catégorie', 'causer', 'cd', 'celer', 'cellule', 'centaine', 'center', 'centers', 'central', 'centre', 'centrer', 'centrer service', 'centric', 'ceo', 'certification', 'certifier', 'cesser', 'cgi', 'cgi favoriser', 'cgi favoriser l', 'cgi plan', 'cgi plan changement', 'cgi reposer', 'cgi reposer taler', 'cgi wwwcgicom', 'cgi wwwcgicom candidature', 'cgi yu', 'cgi yu continuer', 'chain', 'chaine', 'challenge', 'challenge technique', 'challengeant', 'challenger', 'challenging', 'champ', 'chance', 'change', 'changement', 'changement accompagner', 'changement accompagner client', 'changer', 'changing', 'chantier', 'charger', 'charger conception', 'charger développement', 'charger l', 'charger projet', 'chargée', 'charte', 'chasser', 'chaîner', 'chef', 'chef projet', 'chercher', 'chiffrage', 'chiffrer', 'chiffrer affairer', 'chiffrer d', 'chiffrer d affairer', 'choisir', 'choix', 'choix technique', 'chooseyourboss', 'chooseyourboss yu', 'chooseyourboss yu continuer', 'chose', 'cibler', 'cidessous', 'cisco', 'city', 'civil', 'civil military', 'civil military rotorcraft', 'clair', 'class', 'classement', 'classer', 'clean', 'clear', 'clef', 'client', 'client accompagner', 'client acteur', 'client baser', 'client collaborateur', 'client dan', 'client dan démarcher', 'client dan projet', 'client dan secteur', 'client dan transformation', 'client final', 'client grand', 'client grand compter', 'client grandscomptes', 'client grandscomptes dan', 'client l', 'client mission', 'client offrir', 'client partenaire', 'client participer', 'client plaire', 'client profil', 'client proposer', 'client rechercher', 'client secteur', 'client équiper', 'clients', 'clientèle', 'clinical', 'clinique', 'clos', 'closely', 'closing', 'cloud', 'cloud aws', 'cloud azure', 'cloud computing', 'cloud platform', 'cloud public', 'cloud saas', 'cloud saas process', 'clé', 'cms', 'coach', 'coaching', 'codage', 'code', 'coder', 'coding', 'coeur', 'cognos', 'cohérence', 'collaborate', 'collaborateur', 'collaborateur dan', 'collaborateur présent', 'collaborateur présent dan', 'collaborateur répartir', 'collaboratif', 'collaboratifs', 'collaboration', 'collaboration équiper', 'collaborative', 'collaborer', 'collaborer équiper', 'colleagues', 'collect', 'collecter', 'collectif', 'collection', 'collectivement', 'collectivement joignez', 'collectivement joignez prendre', 'collectivité', 'collègue', 'color', 'combattre', 'combattre transport', 'combattre transport mission', 'combiner', 'come', 'comfortable', 'comité', 'commander', 'commencer', 'commerce', 'commercer', 'commercial', 'commerciale', 'commercialisation', 'commercialiser', 'commitment', 'commitment act', 'commitment act integrity', 'committed', 'committed achieving', 'committed achieving workforce', 'commun', 'communauté', 'communauté expert', 'communicant', 'communicate', 'communication', 'communication oral', 'communication skills', 'communiquer', 'community', 'compagnie', 'companies', 'companies helicopters', 'companies helicopters airbus', 'company', 'company’', 'company’ success', 'company’ success reputation', 'compensation', 'competences', 'competencies', 'competitive', 'complet', 'complete', 'complex', 'complexer', 'compliance', 'compliance risks', 'compliance risks commitment', 'complément', 'complément implantation', 'complément implantation proximité', 'complémentaire', 'compléter', 'component', 'comportement', 'composant', 'composer', 'composer technophiles', 'composer technophiles passionner', 'comprehensive', 'comprehensive ranger', 'comprehensive ranger passenger', 'comprendre', 'comprendre besoin', 'compréhension', 'comptabilité', 'comptable', 'compter', 'compter aujourd’hui', 'compter client', 'compter collaborateur', 'compter dan', 'compter plaire', 'comptes', 'computer', 'computer science', 'computing', 'compétence', 'compétence dan', 'compétence développement', 'compétence requis', 'compétence technique', 'compétence égaler', 'compétence égaler poster', 'compétences', 'compétences rechercher', 'compétences requis', 'compétences technique', 'compétitif', 'compétitivité', 'concept', 'concepteur', 'concepteur développeur', 'conception', 'conception développement', 'conception miser', 'conception réalisation', 'conception technique', 'concerner', 'concevoir', 'concevoir développer', 'concevoir réaliser', 'concevoir solution', 'concret', 'concrètement', 'concrétiser', 'concurrentiel', 'condition', 'condition opérationnel', 'conditions', 'conduct', 'conduire', 'conduit', 'conduit changement', 'conduit projet', 'confiance', 'confiance solide', 'confiance solide durable', 'confidentialité', 'confier', 'configuration', 'configurer', 'confluence', 'conformité', 'conformément', 'conférence', 'connaissance', 'connaissance approfondir', 'connaissance baser', 'connaissance d', 'connaissance dan', 'connaissance développement', 'connaissance environnement', 'connaissance l', 'connaissance langage', 'connaissance métier', 'connaissance outil', 'connaissance technique', 'connaissance technologie', 'connaissances', 'connaissances technique', 'connaître', 'connaître période', 'connaître période extraordinaire', 'connect', 'connected', 'connected safer', 'connected safer smarter', 'connecter', 'connexion', 'conseil', 'conseil audit', 'conseil audit user', 'conseil auprès', 'conseil croissance', 'conseil croissance devenir', 'conseil l', 'conseil management', 'conseil service', 'conseil stratégie', 'conseiller', 'consenting', 'consenting airbus', 'consenting airbus storing', 'consister', 'consolidation', 'consolider', 'consommateur', 'consommation', 'constamment', 'constant', 'constant évolution', 'constituer', 'construction', 'construire', 'consultant', 'consultant domaine', 'consultant domaine expertiser', 'consultant expert', 'consultant équiper', 'consultant équiper composer', 'consultante', 'consultants', 'consulter', 'consulting', 'consumer', 'contact', 'contact client', 'contacter', 'container', 'contenir', 'content', 'contenu', 'conter', 'context', 'contexte', 'contexte agile', 'contexte ational', 'contexte mission', 'continent', 'continu', 'continuer', 'continuer accélérer', 'continuer accélérer cgi', 'continuer postuler', 'continuer postuler signaler', 'continuité', 'continuous', 'continuously', 'contract', 'contractuel', 'contrainte', 'contrat', 'contrat pro', 'contrer', 'contribuer', 'contribuer développement', 'contribuer l', 'contribuer projet', 'contribute', 'contribution', 'control', 'contrôle', 'contrôler', 'contrôler gestion', 'convaincre', 'convention', 'conversion', 'conviction', 'convivial', 'convivialité', 'coordinate', 'coordination', 'coordonner', 'core', 'corporate', 'correctif', 'correctif évolutif', 'correction', 'correspondre', 'cost', 'coter', 'coucher', 'countries', 'country', 'couramment', 'courant', 'courbevoie', 'courir', 'couverture', 'couvrir', 'coût', 'create', 'creating', 'creating inclusif', 'creating inclusif working', 'creation', 'creative', 'creativity', 'credit', 'criteo', 'critical', 'critiquer', 'critère', 'critères', 'critères candidat', 'critères candidat niveau', 'crm', 'croire', 'croissance', 'croissance devenir', 'croissance devenir expert', 'croissance entreprendre', 'croissance entreprendre approcher', 'croissance l', 'croissance l plaire', 'cross', 'créatif', 'création', 'créativité', 'crédit', 'crédit agricole', 'créer', 'créer collectivement', 'créer collectivement joignez', 'créé', 'créée', 'cs', 'css', 'css javascript', 'cto', 'cultiver', 'cultural', 'cultural kground', 'cultural kground age', 'culture', 'culture d', 'culture d actionnaire', 'culture propel', 'culture propel accomplish', 'curieux', 'curieux motiver', 'curieux passionner', 'curieuxse', 'curiosité', 'curious', 'current', 'currently', 'cursus', 'customer', 'customer service', 'customer success', 'customers', 'cutting', 'cutting edge', 'cv', 'cv application', 'cv application consenting', 'cv lettre', 'cv lettre motivation', 'cyber', 'cybersécurité', 'cycle', 'cycle vie', 'cérémonie', 'côté', 'cœur', 'd', 'd accompagner', 'd action', 'd actionnaire', 'd actionnaire propriétaire', 'd activité', 'd adaptation', 'd affairer', 'd affairer monder', 'd amélioration', 'd améliorer', 'd analyser', 'd analyser syn', 'd angler', 'd application', 'd architecturer', 'd audit', 'd automatisation', 'd bon', 'd d', 'd dan', 'd développement', 'd emploi', 'd emploi cgi', 'd emploi développeur', 'd emploi yu', 'd entreprendre', 'd entreprise', 'd envergure', 'd environnement', 'd esprit', 'd euro', 'd expert', 'd expertise', 'd expertiser', 'd exploitation', 'd formation', 'd grand', 'd information', 'd informatique', 'd infrastructure', 'd ingénierie', 'd initiative', 'd innovation', 'd intégration', 'd intégration continuer', 'd investissement', 'd min', 'd min requérir', 'd minimum', 'd offrir', 'd outil', 'd poster', 'd professionnel', 'd projet', 'd réussir', 'd significatif', 'd école', 'd école d', 'd écouter', 'd équiper', 'd étude', 'd étude min', 'd évoluer', 'd évolution', 'daily', 'dan', 'dan activité', 'dan ambiance', 'dan aventurer', 'dan boire', 'dan bureau', 'dan cadrer', 'dan cadrer d', 'dan cadrer projet', 'dan cloud', 'dan conception', 'dan conseil', 'dan contexte', 'dan contexte ational', 'dan domaine', 'dan domaine intervention', 'dan domaine l', 'dan définition', 'dan démarcher', 'dan démarcher numérique', 'dan développement', 'dan entreprendre', 'dan environnement', 'dan environnement agile', 'dan environnement ational', 'dan environnement technique', 'dan fonction', 'dan gestion', 'dan grand', 'dan l', 'dan local', 'dan miser', 'dan miser placer', 'dan miser œuvre', 'dan mission', 'dan monder', 'dan métier', 'dan outil', 'dan pays', 'dan phase', 'dan plaire', 'dan plaire pays', 'dan poster', 'dan processus', 'dan projet', 'dan projet lier', 'dan respect', 'dan réalisation', 'dan rôle', 'dan secteur', 'dan secteur banque', 'dan secteur santé', 'dan société', 'dan système', 'dan technologie', 'dan temps', 'dan transformation', 'dan transformation digital', 'dan travail', 'dan univers', 'dan venter', 'dan équiper', 'dan évolution', 'dashboard', 'dashboards', 'data', 'data analyst', 'data analytics', 'data analytics datascience', 'data data', 'data digital', 'data développement', 'data développement informatique', 'data engineer', 'data intelligence', 'data intelligence artificielle', 'data management', 'data science', 'data scientist', 'data scientists', 'data visualisation', 'database', 'databases', 'datacenter', 'dataiku', 'datalake', 'datascience', 'datascience conseil', 'datascience conseil audit', 'dataviz', 'datavizualisation', 'datavizualisation sécurité', 'datavizualisation sécurité systèmes', 'date', 'date début', 'dater', 'dater analytics', 'dater driven', 'dater management', 'dater science', 'dater scientist', 'dater scientists', 'davantage', 'davantage propos', 'davantage propos cgi', 'day', 'day day', 'days', 'db', 'deadlines', 'deal', 'debian', 'decision', 'decision making', 'decisions', 'dedicated', 'deep', 'deep learning', 'define', 'defined', 'definition', 'degree', 'degrees', 'degrees field', 'degrees field study', 'deliver', 'deliverables', 'delivering', 'delivery', 'demain', 'demand', 'demander', 'demander client', 'demonstrate', 'demonstrated', 'department', 'departments', 'deploy', 'deployment', 'descriptif', 'descriptif poster', 'description', 'description l', 'description l offrir', 'description mission', 'description poster', 'design', 'designer', 'designing', 'desired', 'destination', 'destiner', 'detail', 'details', 'determination', 'determination world', 'determination world connected', 'dev', 'develop', 'developer', 'developer poster', 'developers', 'developing', 'development', 'development team', 'developpeur', 'devenir', 'devenir expert', 'devenir expert dan', 'devenir leader', 'devoir', 'devops', 'devops permettre', 'devops permettre ingeniance', 'devoteam', 'diagnostic', 'dialoguer', 'difference', 'diffuser', 'diffusion', 'différence', 'digital', 'digital client', 'digital entreprise', 'digital l', 'digital l accompagnement', 'digital marketing', 'digital transformation', 'digitale', 'digitales', 'digitalisation', 'dimension', 'dimension ationale', 'diplôme', 'diplômer', 'diplômé', 'diplômée', 'diplômée d', 'diplômée école', 'direct', 'directeur', 'direction', 'direction systèmes', 'direction systèmes information', 'direction technique', 'directions', 'directly', 'director', 'directory', 'dirigeant', 'diriger', 'disability', 'disability sexual', 'disability sexual orientation', 'discipliner', 'discover', 'discussion', 'disponibilité', 'disponible', 'disposer', 'disposer d', 'disposer d bon', 'dispositif', 'disposition', 'disruptives', 'distancer', 'distribuer', 'distributed', 'distributeur', 'distribution', 'distribution télécoms', 'distribution télécoms banque', 'diversifier', 'diversity', 'diversity creating', 'diversity creating inclusif', 'diversity teamwork', 'diversity teamwork culture', 'diversité', 'diversité client', 'diversité client grandscomptes', 'division', 'dizaine', 'django', 'docker', 'docker kubernetes', 'document', 'documentaire', 'documentation', 'documentation technique', 'documenter', 'domain', 'domaine', 'domaine d', 'domaine data', 'domaine expertiser', 'domaine expertiser complément', 'domaine intervention', 'domaine intervention ia', 'domaine l', 'donner', 'donner client', 'donner dan', 'donner oracle', 'donner relationnel', 'donner sql', 'donnez', 'donnez élan', 'donnez élan carrière', 'données', 'dossier', 'doter', 'doter d', 'doté', 'dotée', 'dotée d', 'doubler', 'doubler effectif', 'doubler effectif recruter', 'draw', 'draw expertiser', 'draw expertiser experience', 'drive', 'driven', 'driver', 'driving', 'droit', 'drupal', 'dsi', 'durable', 'durable appuyer', 'durable appuyer proximité', 'duration', 'durer', 'durée', 'dynamic', 'dynamics', 'dynamique', 'dynamique stimuler', 'dynamique stimuler l', 'dynamisme', 'dynamisme projet', 'dynamisme projet innovant', 'dysfonctionnement', 'début', 'décision', 'décisionnel', 'découvrez', 'découvrir', 'dédier', 'défense', 'défi', 'défi partager', 'défi partager bénéfice', 'définir', 'définir mettre', 'définir stratégie', 'définition', 'définition miser', 'définition stratégie', 'déjeuner', 'déjà', 'déjà travailler', 'délai', 'délivrer', 'démarcher', 'démarcher numérique', 'démarcher numérique offrir', 'démarrage', 'démarrage asap', 'démontrer', 'département', 'dépasser', 'déplacement', 'déploiement', 'déploiement continu', 'déploiement solution', 'déployer', 'déroulement', 'dérouler', 'détail', 'détailler', 'détecter', 'détection', 'déterminer', 'développement', 'développement activité', 'développement agile', 'développement application', 'développement c', 'développement commercial', 'développement compétence', 'développement d', 'développement d application', 'développement dan', 'développement end', 'développement informatique', 'développement informatique blockchain', 'développement intégration', 'développement java', 'développement k', 'développement l', 'développement logiciel', 'développement maintenance', 'développement nouvelle', 'développement nouvelle fonctionnalité', 'développement outil', 'développement php', 'développement produit', 'développement projet', 'développement rechercher', 'développement réaliser', 'développement service', 'développement solution', 'développement test', 'développement web', 'développements', 'développer', 'développer activité', 'développer application', 'développer compétence', 'développer expertiser', 'développer maintenir', 'développer nouvelle', 'développer nouvelle fonctionnalité', 'développer outil', 'développer solution', 'développeur', 'développeur c', 'développeur end', 'développeur fullstack', 'développeur java', 'développeur net', 'développeur php', 'développeur stack', 'développeur web', 'développeurs', 'développeurse', 'e', 'eau', 'eclipse', 'ecole', 'ecole commerce', 'ecole commercer', 'ecole d', 'ecommerce', 'ecosystem', 'edge', 'education', 'ee', 'effectif', 'effectif recruter', 'effectif recruter consultant', 'effectively', 'effectuer', 'efficace', 'efficacement', 'efficacité', 'efficiency', 'efficient', 'efficient civil', 'efficient civil military', 'effort', 'eg', 'elaboration', 'elaborer', 'elastic', 'elasticsearch', 'electric', 'elk', 'email', 'embarquer', 'embaucher', 'embedded', 'emea', 'emploi', 'emploi cgi', 'emploi cgi yu', 'emploi développeur', 'emploi yu', 'emploi yu signaler', 'employed', 'employed workforce', 'employed workforce airbus', 'employee', 'employees', 'employer', 'employeur', 'employment', 'employment information', 'employment information airbus', 'empower', 'enable', 'encadrement', 'encadrer', 'encourager', 'end', 'end end', 'energie', 'energies', 'energy', 'engagement', 'engagement professionnel', 'engagement professionnel ensemble', 'engager', 'engager dan', 'engie', 'engineer', 'engineering', 'engineering team', 'engineers', 'english', 'english french', 'enjeu', 'enjeu métier', 'enjeu stratégique', 'enjoy', 'enrichir', 'enrichissement', 'enseigner', 'ensemble', 'ensemble projet', 'ensemble relever', 'ensemble relever défi', 'ensemble équiper', 'ensuite', 'ensure', 'ensuring', 'enterprise', 'enthousiasmer', 'enthousiaste', 'entier', 'entities', 'entity', 'entité', 'entité rattachement', 'entraider', 'entreprendre', 'entreprendre approcher', 'entreprendre approcher renforcer', 'entreprendre faire', 'entreprendre faire bon', 'entreprendre leader', 'entreprendre leader marcher', 'entreprendre tailler', 'entreprendre tailler humain', 'entreprendre technology', 'entreprendre technology oriented', 'entrepreneur', 'entrepreneurial', 'entreprise', 'entreprise dan', 'entreprise service', 'entreprise service technologie', 'entreprise services', 'entreprise services numérique', 'entreprises', 'entrer', 'entrer équiper', 'entretenir', 'entretenir partenaire', 'entretenir partenaire relation', 'entretien', 'envergure', 'envie', 'envier', 'envier d', 'environment', 'environment welcome', 'environment welcome application', 'environments', 'environnement', 'environnement agile', 'environnement ational', 'environnement dynamique', 'environnement développement', 'environnement linux', 'environnement production', 'environnement technique', 'environnement technologique', 'environnement travail', 'envisager', 'envoyer', 'envoyer candidature', 'envoyer cv', 'envoyez', 'epm', 'equal', 'equal opportunity', 'equipe', 'equipment', 'equity', 'equivalent', 'ergonomie', 'erp', 'esb', 'esn', 'espace', 'espacer', 'espagne', 'esprit', 'esprit analyser', 'esprit analyser syn', 'esprit d', 'esprit d équiper', 'esprit initiative', 'esprit startup', 'esprit syn', 'esprit équiper', 'essai', 'essential', 'essentiel', 'essentiellement', 'establish', 'estimation', 'estimer', 'etablir', 'etc', 'etes', 'eti', 'etl', 'etude', 'etudes', 'euro', 'euronext', 'europe', 'europe france', 'european', 'european leader', 'european leader providing', 'européen', 'eurovia', 'evaluation', 'event', 'events', 'evolution', 'evolving', 'ex', 'excel', 'excel powerpoint', 'excellence', 'excellence diversity', 'excellence diversity teamwork', 'excellence technologique', 'excellence technologique grouper', 'excellent', 'excellent communication', 'excellente', 'exceller', 'exceller relationnel', 'exceptionnel', 'exchange', 'exciting', 'exclusif', 'execute', 'execution', 'executive', 'executives', 'exemple', 'exercer', 'exhaustif', 'exigeant', 'exigence', 'exiger', 'existant', 'exister', 'existing', 'expansion', 'expect', 'expect donnez', 'expect donnez élan', 'expectations', 'expected', 'expedia', 'experience', 'experience achieve', 'experience achieve excellence', 'experience mobile', 'experience mobile apps', 'experience working', 'experienced', 'experiences', 'expert', 'expert dan', 'expert dan domaine', 'expert technique', 'expertise', 'expertiser', 'expertiser complément', 'expertiser complément implantation', 'expertiser consultant', 'expertiser consultant équiper', 'expertiser dan', 'expertiser experience', 'expertiser experience achieve', 'expertiser multisectorielle', 'expertiser multisectorielle autour', 'expertiser technique', 'experts', 'expliquer', 'exploitation', 'exploiter', 'exploration', 'explorer', 'exposure', 'expression', 'expression besoin', 'exprimer', 'external', 'externe', 'extraction', 'extraordinaire', 'extraordinaire transformation', 'extraordinaire transformation numérique', 'extraordinary', 'extraordinary ground', 'extraordinary ground sky', 'exécuter', 'exécution', 'f', 'fabrication', 'face', 'facebook', 'facilement', 'faciliter', 'facing', 'factory', 'facturation', 'faire', 'faire bon', 'faire bon vivre', 'faire différence', 'faire face', 'faire grandir', 'faire partir', 'faire peur', 'faire preuve', 'faire évoluer', 'faisabilité', 'falloir', 'familial', 'familier', 'famille', 'family', 'fast', 'fast growing', 'fast paced', 'faveur', 'favoriser', 'favoriser l', 'favoriser l équité', 'feature', 'feature team', 'features', 'feedk', 'feel', 'femme', 'ferroviaire', 'feuiller', 'fiabilité', 'fiable', 'fibre', 'ficher', 'fichier', 'fidélisation', 'fidéliser', 'field', 'field study', 'fier', 'filial', 'filiale', 'filière', 'fin', 'fin d', 'fin d étude', 'fin étude', 'final', 'finance', 'finance assurance', 'finance assurance ingeniance', 'finance marché', 'financement', 'financer', 'financer marcher', 'financial', 'financier', 'financier avantgardiste', 'financier avantgardiste technologie', 'find', 'fintech', 'fiscal', 'fixer', 'flexibilité', 'flexible', 'flight', 'flow', 'fluency', 'fluent', 'fluent english', 'fluer', 'flux', 'focus', 'focused', 'foi', 'follow', 'following', 'foncier', 'fonction', 'fonction profil', 'fonction similaire', 'fonctionnalité', 'fonctionnel', 'fonctionnel technique', 'fonctionnement', 'fonctionner', 'fonctions', 'fondamental', 'fondateur', 'fonder', 'fondre', 'fondée', 'force', 'force proposition', 'forcer', 'forcer proposition', 'forfaire', 'formalisation', 'formaliser', 'format', 'formateur', 'formation', 'formation informatique', 'formation justifier', 'formation minimum', 'formation supérieur', 'formation supérieur informatique', 'formation technique', 'formation typer', 'formation universitaire', 'formation école', 'formation équivalent', 'fort', 'fort capacité', 'fort croissance', 'fort culture', 'fort développement', 'fort esprit', 'fort valeur', 'fort valeur ajouter', 'forte', 'fortement', 'fortement apprécier', 'forward', 'foundation', 'foundation company’', 'foundation company’ success', 'fournir', 'fournisseur', 'frai', 'framework', 'frameworks', 'france', 'france ational', 'france iledefrance', 'france l', 'france l ational', 'franchise', 'français', 'français angler', 'frauder', 'free', 'french', 'french english', 'friendly', 'frontend', 'fruit', 'fullstack', 'fully', 'fun', 'function', 'functional', 'functions', 'futur', 'futur employment', 'futur employment information', 'futur ies', 'futur ies responsibilities', 'fédérer', 'g', 'gagner', 'gain', 'game', 'gamme', 'gamme produit', 'garant', 'garantir', 'garantir qualité', 'garder', 'garer', 'gcp', 'gender', 'gender disability', 'gender disability sexual', 'gender identity', 'general', 'generated', 'generated revenu', 'generated revenu €', 'generation', 'gestion', 'gestion actif', 'gestion baser', 'gestion configuration', 'gestion d', 'gestion donner', 'gestion incident', 'gestion processus', 'gestion processus d', 'gestion projet', 'gestion risquer', 'gestionnaire', 'gfi', 'git', 'git jenkins', 'github', 'gitlab', 'global', 'global leader', 'global leader aeronautics', 'goal', 'good', 'good knowledge', 'google', 'google analytics', 'google cloud', 'gouvernance', 'gouvernance donner', 'governance', 'goût', 'goût challenge', 'goût travail', 'goût travail équiper', 'grafana', 'grand', 'grand compter', 'grand compter dan', 'grand distribution', 'grand distribution télécoms', 'grand entreprise', 'grand grouper', 'grand nombre', 'grand projet', 'grand public', 'grande', 'grandir', 'grands', 'grands comptes', 'grandscomptes', 'grandscomptes dan', 'grandscomptes dan secteur', 'graphique', 'graphql', 'great', 'great place', 'great place work', 'gros', 'ground', 'ground sky', 'ground sky space', 'group', 'group société', 'group société conseil', 'groupe', 'grouper', 'grouper ational', 'grouper doubler', 'grouper doubler effectif', 'grow', 'growing', 'growth', 'growth submitting', 'growth submitting cv', 'grâce', 'guidance', 'guidelines', 'guider', 'génie', 'général', 'générale', 'généraliste', 'génération', 'générer', 'géographique', 'gérer', 'gérer projet', 'h', 'hadoop', 'hadoop spark', 'hambourg', 'hambourg république', 'hambourg république tchèque', 'hana', 'handicap', 'handicap yu', 'handicap yu continuer', 'hands', 'happy', 'happy work', 'hard', 'hardware', 'haut', 'haut disponibilité', 'haut niveau', 'hauteur', 'hautsdeseine', 'haver', 'haver experience', 'haver opportunity', 'haver strong', 'having', 'head', 'headquarters', 'health', 'healthcare', 'heart', 'hebdomadaire', 'helicopters', 'helicopters airbus', 'helicopters airbus provides', 'helor', 'help', 'helping', 'helps', 'heure', 'heure continuer', 'heure continuer postuler', 'hibernate', 'high', 'high level', 'high quality', 'highly', 'hip', 'hiring', 'historique', 'hive', 'hoc', 'home', 'homme', 'horaire', 'horizon', 'hp', 'hr', 'html', 'html css', 'html css javascript', 'html javascript', 'hub', 'humain', 'humaines', 'human', 'humeur', 'hybride', 'hébergement', 'héberger', 'hésiter', 'hésiter plaire', 'ia', 'ia big', 'ia big data', 'iaas', 'ibm', 'ici', 'ideal', 'ideally', 'ideas', 'identification', 'identifier', 'identify', 'identifying', 'identity', 'identité', 'idf', 'idéal', 'idéalement', 'idéalement dan', 'idée', 'ie', 'ienne', 'ies', 'ies responsibilities', 'ifrs', 'ihm', 'ile', 'ile france', 'iledefrance', 'im', 'image', 'imager', 'imaginer', 'immobilier', 'impact', 'implantation', 'implantation proximité', 'implantation proximité poursuivre', 'implanter', 'implanter dan', 'implement', 'implementation', 'implementing', 'implication', 'impliquer', 'impliquer dan', 'implémentation', 'implémenter', 'importance', 'important', 'important entreprise', 'important entreprise service', 'importer', 'imposer', 'improve', 'improvement', 'improvements', 'improving', 'impératif', 'impérativement', 'inc', 'incident', 'include', 'includes', 'including', 'inclure', 'inclusif', 'inclusif working', 'inclusif working environment', 'incontournable', 'increase', 'independently', 'indicateur', 'indispensable', 'individual', 'individuals', 'individuel', 'industrial', 'industrialisation', 'industrialiser', 'industrie', 'industriel', 'industriels', 'industriels entretenir', 'industriels entretenir partenaire', 'industry', 'indépendance', 'influencer', 'info', 'info yu', 'info yu continuer', 'info yu jour', 'infogérance', 'informatica', 'information', 'information airbus', 'information airbus airbus', 'information monitoring', 'information monitoring purposes', 'information ti', 'information ti connaître', 'information ti gestion', 'informations', 'informations complémentaire', 'informatique', 'informatique blockchain', 'informatique blockchain philosophie', 'informatique dan', 'informatique gestion', 'informatique justifier', 'informatique justifier d', 'informer', 'infotel', 'infra', 'infrastructure', 'infrastructure cloud', 'infrastructure transport', 'infrastructures', 'infrastructures cloud', 'infrastructures cloud saas', 'ingeniance', 'ingeniance positionner', 'ingeniance positionner entreprendre', 'ingeniance société', 'ingeniance société jeune', 'ingeniance également', 'ingeniance également entreprendre', 'ingénierie', 'initial', 'initiative', 'innovant', 'innovant dan', 'innovant évoluer', 'innovation', 'innovation technologique', 'innovation transformation', 'innovation transformation digital', 'innovative', 'innover', 'inscrire', 'inscrire dan', 'inside', 'insight', 'inspirer', 'installation', 'installer', 'instance', 'institution', 'institutionnel', 'insurance', 'integrated', 'integration', 'integrity', 'integrity foundation', 'integrity foundation company’', 'intellectuel', 'intelligence', 'intelligence artificiel', 'intelligence artificielle', 'intelligence big', 'intelligence big data', 'intelligent', 'interact', 'interactif', 'interaction', 'interagir', 'interest', 'interested', 'interface', 'interlocuteur', 'intermédiaire', 'interpersonal', 'interpersonal skills', 'intervenant', 'intervenir', 'intervenir auprès', 'intervenir client', 'intervenir dan', 'intervenir mission', 'intervenir projet', 'intervention', 'intervention ia', 'intervention ia big', 'interview', 'intitulé', 'intitulé poster', 'intranet', 'intégralité', 'intégrateur', 'intégration', 'intégration continuer', 'intégration dan', 'intégration donner', 'intégration déploiement', 'intégration solution', 'intégration système', 'intégrer', 'intégrer dan', 'intégrer entreprendre', 'intégrer l', 'intégrer l équiper', 'intégrer équiper', 'intégrité', 'intégré', 'intégrée', 'intégrée équiper', 'intéressement', 'intéresser', 'intéressée', 'intérim', 'intérêt', 'investigation', 'investir', 'investir dan', 'investissement', 'investisseur', 'investment', 'inviter', 'involved', 'ios', 'ios android', 'iot', 'ip', 'irrespective', 'irrespective social', 'irrespective social cultural', 'iso', 'issu', 'issu croissance', 'issu croissance entreprendre', 'issue', 'issue d', 'issue d formation', 'issue formation', 'issylesmoulineaux', 'italie', 'itil', 'it’', 'j', 'jamais', 'janvier', 'java', 'java ee', 'java j', 'java javascript', 'java jee', 'java python', 'java spring', 'javascript', 'javascript html', 'javascript html css', 'javascript jquery', 'jboss', 'jee', 'jenkins', 'jeu', 'jeune', 'jeune dynamique', 'jeune dynamique stimuler', 'jira', 'jira confluence', 'job', 'job description', 'job requires', 'job requires awareness', 'joignez', 'joignez prendre', 'joignez prendre partir', 'join', 'joining', 'jouer', 'jouer rôle', 'jour', 'jour adn', 'jour adn excellence', 'jour continuer', 'jour continuer postuler', 'jour signaler', 'jour signaler offrir', 'journey', 'journée', 'jquery', 'js', 'json', 'jump', 'junit', 'juridique', 'jusqu’', 'justifier', 'justifier d', 'justifier d d', 'justifier minimum', 'justifier significatif', 'k', 'k end', 'k office', 'kafka', 'kanban', 'kantar', 'kend', 'kend developer', 'key', 'keyrus', 'kground', 'kground age', 'kground age gender', 'kgrounds', 'kibana', 'klog', 'know', 'knowledge', 'kotlin', 'kpi', 'kpis', 'kubernetes', 'l', 'l accompagnement', 'l accompagnement client', 'l activité', 'l agencer', 'l aider', 'l aise', 'l amélioration', 'l analyser', 'l angler', 'l animation', 'l application', 'l architecturer', 'l assurance', 'l assurance offrir', 'l ational', 'l client', 'l emploi', 'l engagement', 'l engagement professionnel', 'l ensemble', 'l entité', 'l entreprendre', 'l environnement', 'l esprit', 'l esprit d', 'l excellence', 'l expertiser', 'l exploitation', 'l industrie', 'l information', 'l information ti', 'l informatique', 'l infrastructure', 'l ingénierie', 'l innovation', 'l innovation transformation', 'l intégration', 'l intégration solution', 'l objectif', 'l offre', 'l offrir', 'l opportunité', 'l optimisation', 'l oral', 'l organisation', 'l outil', 'l plaire', 'l plaire important', 'l utilisation', 'l écosystème', 'l écouter', 'l écrire', 'l élaboration', 'l énergie', 'l équiper', 'l équiper développement', 'l équité', 'l équité matière', 'l évolution', 'lab', 'laboratoire', 'labsoft', 'lancement', 'lancer', 'langage', 'langage c', 'langage java', 'langage programmation', 'langage python', 'langage sql', 'langages', 'language', 'languages', 'langue', 'langue anglais', 'langues', 'langues anglais', 'laravel', 'large', 'large scale', 'largest', 'law', 'lead', 'lead developer', 'lead développeur', 'leader', 'leader aeronautics', 'leader aeronautics space', 'leader dan', 'leader européen', 'leader marcher', 'leader marcher financier', 'leader mondial', 'leader providing', 'leader providing tanker', 'leadership', 'leading', 'leading space', 'leading space companies', 'leads', 'lean', 'learn', 'learning', 'legal', 'lesjeudiscom', 'lesjeudiscom yu', 'lesjeudiscom yu continuer', 'lettre', 'lettre motivation', 'levalloisperret', 'level', 'levels', 'lever', 'lever fondre', 'leverage', 'levier', 'li', 'liaison', 'librairie', 'libre', 'lien', 'lier', 'lier nouvelle', 'lier nouvelle technologie', 'lieu', 'lieu travail', 'life', 'lifecycle', 'ligne', 'like', 'lille', 'limiter', 'line', 'link', 'linkedin', 'linux', 'linux windows', 'lire', 'liste', 'live', 'livrable', 'livraison', 'livrer', 'll', 'local', 'local ouvrir', 'local ouvrir agencer', 'localisation', 'localisation poster', 'localisation poster localisation', 'located', 'location', 'logiciel', 'logicielle', 'logicielles', 'logique', 'logistique', 'logs', 'loin', 'london', 'londres', 'long', 'long durer', 'long term', 'long terme', 'looking', 'lot', 'lover', 'luire', 'luire permettre', 'lutter', 'luxembourg', 'm', 'mac', 'machine', 'machine learning', 'machiner', 'machiner learning', 'magasin', 'magento', 'mai', 'mai également', 'mail', 'maillage', 'maillage local', 'maillage local ouvrir', 'main', 'maintain', 'maintaining', 'maintenance', 'maintenance application', 'maintenance correctif', 'maintenance correctif évolutif', 'maintenance évolutif', 'maintenir', 'maintien', 'maintien condition', 'maintien condition opérationnel', 'maison', 'maitrise', 'maitriser', 'maitrisez', 'majeur', 'major', 'making', 'manage', 'management', 'management projet', 'management proximité', 'management skills', 'management team', 'management transformation', 'manager', 'manager équiper', 'managers', 'managing', 'managérial', 'mandatory', 'manipulation', 'manière', 'manuel', 'manufacturing', 'maquette', 'marcher', 'marcher financier', 'marcher financier avantgardiste', 'marché', 'mariadb', 'market', 'marketing', 'marketing communication', 'marketing digital', 'marketplace', 'markets', 'maroc', 'marquer', 'marseille', 'marseille recruter', 'marseille recruter hambourg', 'mathématique', 'matière', 'matière d', 'matière d emploi', 'matter', 'matters', 'matériel', 'maven', 'maximum', 'maître', 'maîtrise', 'maîtrise outil', 'maîtriser', 'maîtriser angler', 'maîtriser d', 'maîtriser environnement', 'maîtriser l', 'maîtriser l angler', 'maîtriser langage', 'maîtriser outil', 'maîtriser technologie', 'mba', 'mco', 'mdm', 'means', 'media', 'medical', 'meet', 'meeting', 'meetups', 'meilleur', 'meilleur solution', 'meilleursagents', 'member', 'members', 'membre', 'membre équiper', 'mener', 'mener mission', 'mener projet', 'mensuel', 'mentor', 'message', 'mesurer', 'methodologies', 'methodology', 'methods', 'metrics', 'mettre', 'mettre disposition', 'mettre jour', 'mettre oeuvrer', 'mettre placer', 'mettre œuvre', 'micro', 'microservices', 'microsoft', 'microsoft azure', 'middle', 'middleware', 'mieux', 'migration', 'milieu', 'military', 'military rotorcraft', 'military rotorcraft solution', 'milliard', 'milliard d', 'milliard d euro', 'milliard euro', 'millier', 'million', 'million client', 'million euro', 'min', 'min requérir', 'mind', 'mindset', 'minimum', 'minimum d', 'minimum dan', 'minimum développement', 'minimum poster', 'mining', 'minuter', 'mise', 'mise placer', 'miser', 'miser disposition', 'miser jour', 'miser oeuvrer', 'miser placer', 'miser placer d', 'miser placer solution', 'miser production', 'miser service', 'miser \\x9cuvre', 'miser œuvre', 'miser œuvre projet', 'miser œuvre solution', 'mission', 'mission accepter', 'mission accompagner', 'mission aircraft', 'mission aircraft world', 'mission client', 'mission confier', 'mission conseil', 'mission consister', 'mission d', 'mission développement', 'mission développer', 'mission l', 'mission long', 'mission participer', 'mission principal', 'mission projet', 'mission équiper', 'missions', 'moa', 'mobile', 'mobile apps', 'mobile apps datavizualisation', 'mobiliser', 'mobility', 'mobilité', 'mode', 'mode agile', 'mode projet', 'model', 'modeler', 'modeler donner', 'modeling', 'modelling', 'models', 'modern', 'moderne', 'modernisation', 'modification', 'moduler', 'modélisation', 'modéliser', 'moe', 'moment', 'monder', 'monder entier', 'monder savoir', 'monder savoir davantage', 'mondial', 'mondial dan', 'mongodb', 'monitor', 'monitoring', 'monitoring purposes', 'monitoring purposes relating', 'montage', 'monter', 'monter compétence', 'month', 'monthly', 'months', 'montpellier', 'montrer', 'mot', 'moteur', 'motivated', 'motivation', 'motiver', 'motivée', 'moyen', 'ms', 'multi', 'multiculturel', 'multisectorielle', 'multisectorielle autour', 'multisectorielle autour big', 'mutation', 'mutuel', 'mutuelle', 'mvc', 'mysql', 'mécanique', 'média', 'médical', 'méthode', 'méthode agile', 'méthode agile scrum', 'méthodes', 'méthodique', 'méthodologie', 'méthodologie agile', 'méthodologie agile scrum', 'méthodologique', 'métier', 'métier client', 'métier dan', 'métier l', 'métiers', 'métro', 'm€', 'n', 'nantais', 'nanterre', 'natif', 'national', 'national origin', 'native', 'natixis', 'nature', 'necessary', 'need', 'needed', 'needs', 'negotiation', 'net', 'net c', 'net core', 'network', 'networking', 'networks', 'neuillysurseine', 'new', 'new technologie', 'new york', 'nexus', 'nginx', 'nice', 'niveau', 'niveau angler', 'niveau d', 'niveau d angler', 'niveau d min', 'niveau d étude', 'nlp', 'node', 'nodejs', 'nombre', 'norme', 'nosql', 'noter', 'notion', 'notions', 'notoriété', 'nouveauté', 'nouvelle', 'nouvelle application', 'nouvelle fonctionnalité', 'nouvelle solution', 'nouvelle technologie', 'nouvelle technologie spécialiste', 'novateur', 'numérique', 'numérique esn', 'numérique offrir', 'numérique offrir professionnel', 'numérique organisation', 'numérique organisation continuer', 'numéro', 'nécessaire', 'nécessiter', 'négociation', 'négocier', 'object', 'objectif', 'objectiver', 'objet', 'objet connecter', 'objets', 'obligatoire', 'obtenir', 'occasion', 'occuper', 'octobre', 'oeuvrer', 'offer', 'offers', 'offers comprehensive', 'offers comprehensive ranger', 'office', 'officer', 'offre', 'offrir', 'offrir aujourd’hui', 'offrir aujourd’hui équiper', 'offrir client', 'offrir d', 'offrir d emploi', 'offrir expertiser', 'offrir expertiser multisectorielle', 'offrir professionnel', 'offrir professionnel opportunité', 'offrir service', 'offshore', 'onboarding', 'online', 'open', 'open source', 'opensource', 'openstack', 'operating', 'operational', 'operations', 'opportunities', 'opportunity', 'opportunity employer', 'opportunité', 'opportunité carrière', 'opportunité carrière stimulant', 'ops', 'optimal', 'optimisation', 'optimisation performance', 'optimiser', 'optimization', 'optimize', 'option', 'optique', 'opérateur', 'opération', 'opérationnel', 'opérations', 'opérer', 'oracle', 'oral', 'oral écrire', 'oral écrit', 'orange', 'orchestration', 'order', 'ordre', 'organisation', 'organisation continuer', 'organisation continuer accélérer', 'organisationnel', 'organiser', 'organisme', 'organisée', 'organization', 'organizational', 'organizations', 'orientation', 'orientation religious', 'orientation religious belief', 'oriented', 'oriented offrir', 'oriented offrir expertiser', 'orienter', 'orienter objet', 'origin', 'original', 'origine', 'os', 'ouest', 'oui', 'outcomes', 'outil', 'outil bi', 'outil bureautique', 'outil d', 'outil développement', 'outil gestion', 'outil informatique', 'outillage', 'outils', 'ouverture', 'ouverture esprit', 'ouvrage', 'ouvrager', 'ouvrir', 'ouvrir agencer', 'ouvrir agencer aix', 'ouvrir situation', 'ouvrir situation handicap', 'overall', 'owner', 'owners', 'ownership', 'paas', 'paced', 'pack', 'pack office', 'package', 'packaging', 'page', 'paiement', 'pair', 'palette', 'palmarès', 'panel', 'parallèle', 'paramétrage', 'parc', 'parcourir', 'parcourir client', 'parfait', 'parfaitement', 'paribas', 'partager', 'partager bénéfice', 'partager bénéfice issu', 'partager connaissance', 'partager valeur', 'partenaire', 'partenaire relation', 'partenaire relation confiance', 'partenariat', 'participate', 'participation', 'participer', 'participer activement', 'participer amélioration', 'participer conception', 'participer définition', 'participer développement', 'participer l', 'participer miser', 'participer miser placer', 'participer phase', 'participer projet', 'partiel', 'partir', 'partir croissance', 'partir croissance l', 'partir pre', 'partner', 'partners', 'partnership', 'partout', 'party', 'passage', 'passenger', 'passenger airliners', 'passenger airliners airbus', 'passer', 'passion', 'passion determination', 'passion determination world', 'passionate', 'passionnant', 'passionner', 'passionner renforcer', 'passionner renforcer jour', 'passionné', 'passionnée', 'patient', 'patrimoine', 'patterns', 'pay', 'payer', 'pays', 'pc', 'pendre', 'penser', 'people', 'people work', 'people work passion', 'perform', 'performance', 'performant', 'perl', 'permanence', 'permanent', 'permanenter', 'permettre', 'permettre d', 'permettre développer', 'permettre ingeniance', 'permettre ingeniance positionner', 'perpétuel', 'person', 'personal', 'personal dater', 'personnaliser', 'personnalité', 'personnel', 'perspectif', 'perspective', 'perspective évolution', 'perspective évolution profil', 'pertinence', 'pertinent', 'petit', 'petit équiper', 'peur', 'pharmaceutique', 'phase', 'phase conception', 'phase d', 'phase projet', 'phase test', 'philosophie', 'philosophie devops', 'philosophie devops permettre', 'phone', 'php', 'php mysql', 'php symfony', 'physique', 'pilier', 'pilotage', 'pilotage performance', 'pilotage projet', 'piloter', 'piloter projet', 'pipeline', 'pl', 'pl sql', 'place', 'place work', 'placer', 'placer d', 'placer outil', 'placer solution', 'placer taking', 'placer taking pride', 'placer test', 'plaire', 'plaire adapter', 'plaire client', 'plaire collaborateur', 'plaire connaissance', 'plaire d', 'plaire dan', 'plaire grand', 'plaire important', 'plaire important entreprise', 'plaire innovant', 'plaire million', 'plaire pays', 'plaire performant', 'plaire plaire', 'plaire poster', 'plaire rejoindre', 'plaire secret', 'plaisir', 'plan', 'plan action', 'plan changement', 'plan changement accompagner', 'plan d', 'plan test', 'planification', 'planifier', 'planning', 'plateforme', 'platform', 'platforms', 'play', 'player', 'pleinement', 'plm', 'pluridisciplinaire', 'pme', 'po', 'poc', 'poindre', 'point', 'pointer', 'pointu', 'policies', 'policy', 'politique', 'polyvalence', 'polyvalent', 'ponctuel', 'portail', 'portefeuille', 'portefeuille client', 'porter', 'porteur', 'portfolio', 'positif', 'position', 'position description', 'positionnement', 'positionner', 'positionner entreprendre', 'positionner entreprendre leader', 'possibilité', 'possibilité d', 'posséder', 'posséder bon', 'post', 'poste', 'poste baser', 'poste mission', 'poste pourvoir', 'poster', 'poster baser', 'poster développeur', 'poster faire', 'poster localisation', 'poster localisation poster', 'poster ouvrir', 'poster ouvrir situation', 'poster pourvoir', 'poster similaire', 'postgresql', 'postuler', 'postuler offrir', 'postuler signaler', 'postuler signaler offrir', 'postuler yu', 'postuler yu continuer', 'postulez', 'potential', 'potential compliance', 'potential compliance risks', 'potentiel', 'poursuivre', 'poursuivre maillage', 'poursuivre maillage local', 'pourvoir', 'pousser', 'pouvoir', 'pouvoir amener', 'pouvoir également', 'power', 'power bi', 'powerbi', 'powerpoint', 'powershell', 'practice', 'practices', 'pragmatique', 'pratiquer', 'pratiquer développement', 'pre', 'preferred', 'premise', 'prendre', 'prendre charger', 'prendre compter', 'prendre initiative', 'prendre partir', 'prendre partir croissance', 'preparation', 'prepare', 'presence', 'present', 'presentation', 'presentations', 'prestataire', 'prestation', 'prestation service', 'prestigieux', 'preuve', 'preuve d', 'previous', 'pricing', 'pride', 'pride work', 'pride work draw', 'primary', 'prime', 'primer', 'principal', 'principal acteur', 'principal mission', 'principal mission accepter', 'principalement', 'principales', 'principe', 'principles', 'prioriser', 'priorities', 'priorité', 'prise', 'priser', 'priser charger', 'priser décision', 'privacy', 'private', 'priver', 'privilégier', 'prix', 'pro', 'proactif', 'proactive', 'proactivité', 'problem', 'problem solving', 'problems', 'problème', 'problématique', 'procedures', 'process', 'process industriels', 'process industriels entretenir', 'process recrutement', 'processes', 'processing', 'processus', 'processus d', 'processus d affairer', 'processus métier', 'processus recrutement', 'prochain', 'procurement', 'procéder', 'procédure', 'product', 'product management', 'product manager', 'product owner', 'product team', 'production', 'productivité', 'products', 'produire', 'produit', 'produit service', 'produits', 'professional', 'professionals', 'professionnalisme', 'professionnel', 'professionnel bénéficier', 'professionnel bénéficier valeur', 'professionnel dan', 'professionnel ensemble', 'professionnel ensemble relever', 'professionnel opportunité', 'professionnel opportunité carrière', 'proficiency', 'profil', 'profil candidat', 'profil compétence', 'profil formation', 'profil formation informatique', 'profil formation supérieur', 'profil issue', 'profil rechercher', 'profil rechercher diplômée', 'profil rechercher formation', 'profil rechercher profil', 'profil recherché', 'profil requis', 'profile', 'profiler', 'profit', 'profiter', 'progiciel', 'progiciels', 'program', 'programmation', 'programmer', 'programming', 'programs', 'progress', 'progresser', 'progression', 'project', 'project management', 'project manager', 'project team', 'projects', 'projet', 'projet agile', 'projet ambitieux', 'projet client', 'projet d', 'projet dan', 'projet digital', 'projet développement', 'projet fort', 'projet fort valeur', 'projet informatique', 'projet innovant', 'projet innovant évoluer', 'projet lier', 'projet lier nouvelle', 'projet miser', 'projet mission', 'projet participer', 'projet plaire', 'projet professionnel', 'projet profil', 'projet stratégique', 'projet technique', 'projet transformation', 'projet web', 'projet équiper', 'projets', 'promote', 'promoteur', 'promotion', 'promouvoir', 'prononcer', 'propel', 'propel accomplish', 'propel accomplish extraordinary', 'propos', 'propos cgi', 'propos cgi wwwcgicom', 'proposer', 'proposer client', 'proposer rejoindre', 'proposer solution', 'proposition', 'proposition commercial', 'propre', 'propriétaire', 'propriétaire professionnel', 'propriétaire professionnel bénéficier', 'prospecter', 'prospection', 'prospects', 'protection', 'protection donner', 'protocole', 'prototype', 'protéger', 'proud', 'proven', 'provenir', 'provenir cabinet', 'provenir cabinet recrutement', 'provide', 'provider', 'providers', 'provides', 'provides efficient', 'provides efficient civil', 'providing', 'providing tanker', 'providing tanker combattre', 'proximité', 'proximité poursuivre', 'proximité poursuivre maillage', 'proximité équiper', 'proximité équiper expertiser', 'préciser', 'précisément', 'préconisations', 'précédent', 'prédictif', 'préembauche', 'préférence', 'préférer', 'préoccupation', 'préparation', 'préparer', 'prérequis', 'présence', 'présent', 'présent dan', 'présent dan pays', 'présentation', 'présenter', 'prévision', 'prévoir', 'prévoyance', 'prêt', 'prête', 'prôner', 'public', 'publication', 'publicitaire', 'puissance', 'puppet', 'purpose', 'purposes', 'purposes relating', 'purposes relating application', 'push', 'puteaux', 'pwc', 'python', 'python java', 'python r', 'pédagogie', 'pédagogique', 'pédagogue', 'pérenne', 'pérennité', 'périmètre', 'période', 'période extraordinaire', 'période extraordinaire transformation', 'pôle', 'qa', 'qlik', 'qlikview', 'qu', 'qualification', 'qualification successful', 'qualification successful role', 'qualifications', 'qualified', 'qualified applicants', 'qualifier', 'qualitatif', 'quality', 'qualité', 'qualité coder', 'qualité donner', 'qualité livrable', 'qualité relationnel', 'qualité rédactionnel', 'qualité service', 'qualités', 'qualités requis', 'quantitatif', 'question', 'quickly', 'quo', 'quotidien', 'quotidiennement', 'r', 'r looking', 'r python', 'rabbitmq', 'race', 'race color', 'radio', 'rails', 'raison', 'ranger', 'ranger passenger', 'ranger passenger airliners', 'rapide', 'rapidement', 'rapport', 'rassembler', 'rattachement', 'rattacher', 'rattaché', 'rattachée', 'rattachée responsable', 'rd', 'reach', 'react', 'react native', 'reactjs', 'ready', 'real', 'real time', 'receive', 'recette', 'recevoir', 'recherche', 'rechercher', 'rechercher actuellement', 'rechercher client', 'rechercher compétences', 'rechercher consultant', 'rechercher d', 'rechercher diplômée', 'rechercher diplômée d', 'rechercher développement', 'rechercher développeur', 'rechercher formation', 'rechercher formation supérieur', 'rechercher profil', 'rechercher une', 'recherché', 'recognized', 'recommandation', 'recommendation', 'reconnaissance', 'reconnaître', 'reconnaître dan', 'reconnu', 'reconnue', 'record', 'recruiting', 'recruitment', 'recrutement', 'recrutement retenue', 'recrutement retenue cgi', 'recrutement spécialiser', 'recruter', 'recruter consultant', 'recruter consultant domaine', 'recruter développeur', 'recruter hambourg', 'recruter hambourg république', 'recruter une', 'recruteur', 'recueil', 'recueil besoin', 'recueillir', 'recul', 'redhat', 'redis', 'redux', 'refonte', 'regard', 'region', 'regional', 'regrouper', 'regular', 'regulatory', 'rejoignez', 'rejoignez équiper', 'rejoindre', 'rejoindre communauté', 'rejoindre entreprendre', 'rejoindre intégrer', 'rejoindre l', 'rejoindre équiper', 'related', 'related service', 'related service generated', 'relatif', 'relating', 'relating application', 'relating application futur', 'relation', 'relation client', 'relation confiance', 'relation confiance solide', 'relationnel', 'relationnel capacité', 'relationnel sentir', 'relationship', 'relationships', 'release', 'relever', 'relever challenge', 'relever défi', 'relever défi partager', 'reliability', 'religion', 'religious', 'religious belief', 'religious belief yu', 'remboursement', 'remonter', 'remote', 'remplir', 'rencontrer', 'rendezvous', 'renforcer', 'renforcer culture', 'renforcer culture d', 'renforcer jour', 'renforcer jour adn', 'renforcer pôle', 'renforcer équiper', 'rennes', 'renouvelable', 'rentabilité', 'report', 'reporting', 'reportings', 'reposer', 'reposer taler', 'reposer taler l', 'représenter', 'reputation', 'reputation sustainable', 'reputation sustainable growth', 'request', 'requests', 'required', 'required qualification', 'required qualification successful', 'required skills', 'requirements', 'requires', 'requires awareness', 'requires awareness potential', 'requis', 'requis formation', 'requis yu', 'requis yu signaler', 'requérir', 'requête', 'research', 'resources', 'respect', 'respect délai', 'respect norme', 'respecter', 'responsabilité', 'responsabilités', 'responsable', 'responsibilities', 'responsibility', 'responsible', 'responsive', 'ressource', 'ressource humain', 'ressources', 'ressources humaines', 'rest', 'restaurant', 'restauration', 'restaurer', 'rester', 'restitution', 'results', 'retail', 'retenue', 'retenue cgi', 'retenue cgi favoriser', 'retrouver', 'retrouver ambiance', 'revenu', 'revenu €', 'revenu € billion', 'review', 'reviews', 'revu', 'revue', 'revue coder', 'rgpd', 'rh', 'riche', 'right', 'rigoureux', 'rigoureux autonome', 'rigoureuxse', 'rigueur', 'rigueur autonomie', 'rigueur organisation', 'risk', 'risk management', 'risks', 'risks commitment', 'risks commitment act', 'risquer', 'risques', 'roadmap', 'robotique', 'robuste', 'robustesse', 'roi', 'role', 'roles', 'rotorcraft', 'rotorcraft solution', 'rotorcraft solution worldwide', 'router', 'rtt', 'ruby', 'ruer', 'run', 'running', 'réactif', 'réactivité', 'réalisation', 'réalisation d', 'réalisation développement', 'réalisation projet', 'réalisation test', 'réaliser', 'réaliser chiffrer', 'réaliser développement', 'réaliser mission', 'réaliser projet', 'réaliser test', 'réaliser étude', 'réalité', 'récent', 'rédaction', 'rédaction documentation', 'rédaction spécification', 'rédaction spécification fonctionnel', 'rédaction spécification technique', 'rédactionnel', 'rédiger', 'rédiger documentation', 'rédiger documentation technique', 'rédiger spécification', 'rédiger spécification technique', 'réel', 'réflexion', 'référence', 'référencement', 'référent', 'référent technique', 'référentiel', 'région', 'région ienne', 'régional', 'réglementaire', 'réglementation', 'régler', 'régression', 'régulier', 'régulièrement', 'réinventer', 'rémunération', 'rémunération profil', 'répartir', 'répartir dan', 'répondre', 'répondre besoin', 'répondre enjeu', 'réponse', 'réponse appel', 'réponse appel offrir', 'république', 'république tchèque', 'république tchèque suisse', 'réseau', 'réseau social', 'réseau sécurité', 'réseaux', 'résolument', 'résolution', 'résolution problème', 'résoudre', 'résoudre problème', 'résultat', 'réunion', 'réunir', 'réussir', 'réussir dan', 'réussir poster', 'réussite', 'réussite cgi', 'réussite cgi reposer', 'révolution', 'révolution numérique', 'révolutionner', 'rêver', 'rôle', 's', 's hana', 'saas', 'saas process', 'saas process industriels', 'safe', 'safer', 'safer smarter', 'safer smarter placer', 'safety', 'safran', 'saint', 'saisir', 'salarier', 'salary', 'saler', 'saler team', 'sales', 'salesforce', 'salesforcecom', 'salle', 'salon', 'santé', 'santé transport', 'santé transport grand', 'sap', 'sas', 'sas airbus', 'sas airbus global', 'sass', 'satellite', 'satisfaction', 'satisfaction client', 'sauvegarder', 'savoir', 'savoir accompagner', 'savoir accompagner évolution', 'savoir davantage', 'savoir davantage propos', 'savoir faire', 'savoir faire preuve', 'savoir plaire', 'savoir travailler', 'savoirfaire', 'scala', 'scalabilité', 'scalable', 'scale', 'school', 'schéma', 'science', 'sciences', 'scientific', 'scientifique', 'scientist', 'scientists', 'scope', 'scratch', 'script', 'scripting', 'scrum', 'scrum kanban', 'scénario', 'search', 'second', 'secret', 'secteur', 'secteur activité', 'secteur banque', 'secteur banque finance', 'secteur d', 'secteur d activité', 'secteur l', 'secteur public', 'secteur santé', 'secteur santé transport', 'secteur technologie', 'secteur technologie l', 'sector', 'sectoriel', 'secure', 'security', 'seeking', 'segment', 'selenium', 'self', 'selling', 'sens', 'sense', 'sensibilité', 'sensible', 'sentir', 'sentir l', 'sentir organisation', 'sentir relationnel', 'sentir service', 'sentir service client', 'sentir travail', 'sentir travail équiper', 'seo', 'septembre', 'serez', 'serez activus', 'serez activus activus', 'server', 'serveur', 'service', 'service client', 'service cloud', 'service d', 'service dan', 'service generated', 'service generated revenu', 'service informatique', 'service numérique', 'service solution', 'service technologie', 'service technologie l', 'services', 'services numérique', 'servir', 'session', 'set', 'sex', 'sexual', 'sexual orientation', 'sexual orientation religious', 'sgbd', 'shape', 'share', 'sharepoint', 'sharing', 'shell', 'sig', 'signaler', 'signaler offrir', 'signaler offrir d', 'signature', 'significant', 'significatif', 'significatif dan', 'significatif développement', 'similaire', 'similaire requis', 'similaire requis yu', 'similaire souhaité', 'similaire souhaité yu', 'similar', 'simple', 'simplicité', 'simplifier', 'simulation', 'site', 'site web', 'situation', 'situation handicap', 'situer', 'siéger', 'skills', 'skills ability', 'skills experience', 'sky', 'sky space', 'sky space job', 'small', 'smart', 'smarter', 'smarter placer', 'smarter placer taking', 'snef', 'soa', 'soap', 'social', 'social cultural', 'social cultural kground', 'social media', 'société', 'société conseil', 'société conseil croissance', 'société jeune', 'société jeune dynamique', 'société service', 'socle', 'socle technique', 'soft', 'soft skills', 'software', 'software development', 'soif', 'soin', 'soirée', 'solid', 'solide', 'solide connaissance', 'solide durable', 'solide durable appuyer', 'solliciter', 'solliciter provenir', 'solliciter provenir cabinet', 'solution', 'solution adapter', 'solution cloud', 'solution d', 'solution gestion', 'solution innovant', 'solution logicielles', 'solution technique', 'solution technologique', 'solution worldwide', 'solution worldwide people', 'solutions', 'solve', 'solving', 'sommer', 'sommer plaire', 'sommer rechercher', 'sommes', 'sonar', 'sortir', 'sou', 'sou responsabilité', 'souci', 'soucieux', 'souder', 'souhait', 'souhaiter', 'souhaiter faire', 'souhaiter intégrer', 'souhaiter investir', 'souhaiter participer', 'souhaiter rejoindre', 'souhaiter retrouver', 'souhaiter retrouver ambiance', 'souhaiter travailler', 'souhaiter évoluer', 'souhaité', 'souhaité yu', 'souhaité yu signaler', 'source', 'source donner', 'sourcing', 'soutenir', 'soutien', 'space', 'space companies', 'space companies helicopters', 'space job', 'space job description', 'space related', 'space related service', 'spark', 'spatial', 'speak', 'specialist', 'specific', 'specifications', 'specified', 'spirit', 'splunk', 'spoken', 'sport', 'sportif', 'spring', 'spring boot', 'spring hibernate', 'sprint', 'spécialisation', 'spécialiser', 'spécialiser dan', 'spécialiste', 'spécialiste secteur', 'spécialiste secteur banque', 'spécialité', 'spécification', 'spécification fonctionnel', 'spécification fonctionnel technique', 'spécification technique', 'spécifique', 'sql', 'sql nosql', 'sql server', 'squad', 'ssii', 'ssis', 'ssrs', 'stabilité', 'stack', 'stack technique', 'staff', 'stakeholders', 'standard', 'start', 'start ups', 'startup', 'startups', 'stater', 'statistical', 'statistics', 'statistique', 'status', 'statut', 'stimulant', 'stimulant réussite', 'stimulant réussite cgi', 'stimuler', 'stimuler l', 'stimuler l innovation', 'stock', 'stockage', 'stocker', 'storage', 'store', 'stories', 'storing', 'storing information', 'storing information monitoring', 'story', 'strategic', 'strategies', 'strategy', 'stratégie', 'stratégie digital', 'stratégique', 'streaming', 'stress', 'strive', 'strong', 'structuration', 'structured', 'structurer', 'structurer tailler', 'structurer tailler humain', 'studies', 'studio', 'study', 'subject', 'submitting', 'submitting cv', 'submitting cv application', 'success', 'success reputation', 'success reputation sustainable', 'success story', 'successful', 'successful role', 'succès', 'sud', 'suisse', 'suisse type', 'suisse type d', 'suite', 'suivi', 'sujet', 'summary', 'super', 'superviser', 'supervision', 'suppliers', 'supply', 'supply chain', 'supplémentaire', 'support', 'support niveau', 'support technique', 'support utilisateur', 'support équiper', 'supporter', 'supporting', 'supérieur', 'supérieur informatique', 'surmesure', 'surveillance', 'sustainable', 'sustainable growth', 'sustainable growth submitting', 'svn', 'swift', 'symfony', 'sympa', 'syn', 'synergie', 'system', 'systems', 'système', 'système d', 'système d information', 'système information', 'système linux', 'système réseau', 'systèmes', 'systèmes d', 'systèmes d information', 'systèmes information', 'systèmes infrastructures', 'systèmes infrastructures cloud', 'sécurisation', 'sécuriser', 'sécurité', 'sécurité système', 'sécurité systèmes', 'sécurité systèmes infrastructures', 'sélection', 'sélectionner', 'séminaire', 'sérieux', 'sûr', 't', 'tableau', 'tableau bord', 'tabler', 'tablette', 'tailler', 'tailler humain', 'taire', 'taire aimer', 'taire pouvoir', 'takes', 'taking', 'taking pride', 'taking pride work', 'talend', 'talent', 'talented', 'talents', 'talentueux', 'taler', 'taler l', 'taler l engagement', 'tanker', 'tanker combattre', 'tanker combattre transport', 'target', 'targets', 'tasks', 'taux', 'tchèque', 'tchèque suisse', 'tchèque suisse type', 'tdd', 'team', 'team building', 'team members', 'team player', 'team work', 'teams', 'teamwork', 'teamwork culture', 'teamwork culture propel', 'tech', 'tech lead', 'tech valley', 'technical', 'technicien', 'technico', 'technico fonctionnel', 'technique', 'technique client', 'technique d', 'technique dan', 'technique développement', 'technique fonctionnel', 'technique java', 'technique l', 'technique participer', 'technique profil', 'technique projet', 'technique requis', 'technique solution', 'technique équiper', 'techniquement', 'techniques', 'techno', 'technologie', 'technologie big', 'technologie disruptives', 'technologie information', 'technologie information ti', 'technologie innovant', 'technologie java', 'technologie l', 'technologie l information', 'technologie microsoft', 'technologie souhaiter', 'technologie spécialiste', 'technologie spécialiste secteur', 'technologie web', 'technologies', 'technologique', 'technologique grouper', 'technologique grouper doubler', 'technology', 'technology oriented', 'technology oriented offrir', 'technophiles', 'technophiles passionner', 'technophiles passionner renforcer', 'technos', 'telecom', 'temps', 'temps réel', 'temps travail', 'tempérament', 'tendance', 'tenu', 'term', 'terme', 'terraform', 'terrain', 'territoire', 'territory', 'tertiaire', 'test', 'test automatiser', 'test fonctionnel', 'test intégration', 'test recette', 'test technique', 'test unitaire', 'tester', 'testing', 'tests', 'tfs', 'thales', 'things', 'think', 'thinking', 'thrive', 'thématique', 'ti', 'ti connaître', 'ti connaître période', 'ti gestion', 'ti gestion processus', 'ticket', 'ticket restaurer', 'tickets', 'tiers', 'time', 'timely', 'tirer', 'titrer', 'titulaire', 'titulaire d', 'tma', 'today', 'tomcat', 'tool', 'tools', 'topics', 'total', 'toucher', 'tour', 'tourner', 'tourner ver', 'track', 'track record', 'tracking', 'trade', 'trading', 'traduire', 'traffic', 'trafic', 'train', 'training', 'traitement', 'traitement donner', 'traiter', 'transaction', 'transfert', 'transform', 'transformation', 'transformation digital', 'transformation digital entreprise', 'transformation digital l', 'transformation digitale', 'transformation métier', 'transformation numérique', 'transformation numérique organisation', 'transformer', 'transition', 'transition énergétique', 'transmettre', 'transmission', 'transparence', 'transparent', 'transport', 'transport grand', 'transport grand distribution', 'transport mission', 'transport mission aircraft', 'transversal', 'transverse', 'transverses', 'travail', 'travail collaboratif', 'travail d', 'travail équiper', 'travailler', 'travailler collaboration', 'travailler dan', 'travailler dan environnement', 'travailler mode', 'travailler projet', 'travailler équiper', 'travailler étroit', 'travailler étroit collaboration', 'travailleur', 'travel', 'travers', 'travers monder', 'trends', 'tribu', 'trouver', 'trust', 'trusted', 'twitter', 'type', 'type contrat', 'type d', 'type d emploi', 'typer', 'typer école', 'types', 'typescript', 'tâcher', 'télécom', 'télécommunication', 'télécoms', 'télécoms banque', 'télécoms banque l', 'téléphoner', 'téléphonique', 'télétravail', 'ui', 'uk', 'umanis', 'uml', 'understand', 'understanding', 'une', 'une développeur', 'uniquement', 'unir', 'unit', 'unitaire', 'unitaire intégration', 'units', 'unité', 'univers', 'universitaire', 'university', 'université', 'uniware', 'unix', 'unix linux', 'ups', 'urbain', 'usa', 'usage', 'user', 'user caser', 'user experience', 'user experience mobile', 'user stories', 'users', 'usiner', 'utile', 'utilisateur', 'utilisation', 'utilisation outil', 'utiliser', 'utiliser technologie', 'utilities', 'ux', 'ux ui', 'v', 'valeur', 'valeur ajouter', 'valeur créer', 'valeur créer collectivement', 'valeur humain', 'validation', 'valider', 'valley', 'valoir', 'valorisation', 'valoriser', 'variable', 'varier', 'variety', 'vba', 've', 'veille', 'veille technologique', 'veiller', 'veiller technologique', 'vendor', 'vendre', 'venez', 'venir', 'vente', 'venter', 'ver', 'ver cloud', 'verbal', 'verbal written', 'version', 'veteran', 'veteran status', 'video', 'vidéo', 'vie', 'view', 'vigueur', 'ville', 'vinci', 'vinci energies', 'virtualisation', 'virtuel', 'viser', 'visibilité', 'vision', 'visiter', 'visual', 'visual studio', 'visualisation', 'visuel', 'visàvis', 'vite', 'vivre', 'vmware', 'vocation', 'voir', 'voire', 'volet', 'volonté', 'volume', 'vouloir', 'voyager', 'vrai', 'vue', 'vuejs', 'véhiculer', 'vérification', 'vérifier', 'véritable', 'want', 'way', 'ways', 'web', 'web application', 'web mobile', 'web service', 'web services', 'webpack', 'webservices', 'website', 'welcome', 'welcome application', 'welcome application irrespective', 'we’', 'we’ r', 'wide', 'willing', 'windows', 'windows linux', 'word', 'wordpress', 'work', 'work closely', 'work draw', 'work draw expertiser', 'work environment', 'work passion', 'work passion determination', 'workflow', 'workflows', 'workforce', 'workforce airbus', 'workforce airbus offers', 'workforce diversity', 'workforce diversity creating', 'working', 'working environment', 'working environment welcome', 'workplace', 'works', 'workshops', 'world', 'world connected', 'world connected safer', 'world leading', 'world leading space', 'worldline', 'worldwide', 'worldwide people', 'worldwide people work', 'world’', 'wpf', 'write', 'writing', 'written', 'written communication', 'wwwcgicom', 'wwwcgicom candidature', 'wwwcgicom candidature solliciter', 'x', 'xml', 'year', 'years', 'years experience', 'york', 'you’', 'you’ ll', 'you’ r', 'yu', 'yu comprendre', 'yu continuer', 'yu continuer postuler', 'yu heure', 'yu heure continuer', 'yu jour', 'yu jour continuer', 'yu jour signaler', 'yu signaler', 'yu signaler offrir', 'z', 'zone', '\\x9cuvre', '«', '®', '°', '»', '» «', 'âme', 'échanger', 'échelle', 'école', 'école commercer', 'école d', 'économie', 'économique', 'écosystème', 'écouter', 'écran', 'écrire', 'écrire oral', 'écrit', 'écrit oral', 'écriture', 'éditer', 'éditeur', 'éditeur logiciel', 'édition', 'également', 'également entreprendre', 'également entreprendre technology', 'égaler', 'égaler poster', 'égaler poster ouvrir', 'égalité', 'élaboration', 'élaborer', 'élan', 'élan carrière', 'élan carrière secteur', 'élargir', 'électrique', 'électronique', 'élever', 'élément', 'énergie', 'énergétique', 'épanouir', 'épanouir dan', 'épanouissement', 'équilibrer', 'équipement', 'équiper', 'équiper accompagner', 'équiper agile', 'équiper bon', 'équiper charger', 'équiper client', 'équiper commercial', 'équiper composer', 'équiper composer technophiles', 'équiper consultant', 'équiper d', 'équiper dan', 'équiper data', 'équiper dynamique', 'équiper dynamisme', 'équiper développement', 'équiper développeurs', 'équiper expert', 'équiper expertiser', 'équiper expertiser consultant', 'équiper jeune', 'équiper jeune dynamique', 'équiper l', 'équiper marketing', 'équiper métier', 'équiper passionner', 'équiper permettre', 'équiper perspective', 'équiper perspective évolution', 'équiper pluridisciplinaire', 'équiper production', 'équiper produire', 'équiper profil', 'équiper projet', 'équiper rd', 'équiper rechercher', 'équiper sentir', 'équiper support', 'équiper tailler', 'équiper tailler humain', 'équiper technique', 'équiper travailler', 'équité', 'équité matière', 'équité matière d', 'équivalent', 'équivaloir', 'établir', 'établissement', 'étape', 'état', 'état esprit', 'étendre', 'éthique', 'étranger', 'étroit', 'étroit collaboration', 'étroit collaboration équiper', 'étude', 'étude développement', 'étude min', 'étude min requérir', 'études', 'étudiant', 'étudier', 'évaluation', 'évaluer', 'éventuel', 'éventuellement', 'évoluer', 'évoluer dan', 'évoluer dan environnement', 'évoluer ver', 'évolutif', 'évolution', 'évolution carrière', 'évolution professionnel', 'évolution profil', 'évènement', 'événement', 'œuvre', 'œuvre projet', 'œuvre solution', '–', '‘', '“', '”', '•', '…', '… connaissance', '… participer', '… profil', '€', '€ billion', '€ billion employed', '\\uf0a7', '\\uf0b7', '\\uf0d8', '\\uf0fc']"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "print(features_description)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### IV. Find the (features x topics) matric with the LDA/LSI/NMF and generate groups\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Une fois nos features extraites, on choisit d'utiliser trois méthodes pour extraire les topics: Nonnegative Matrix Factorisation, Latent Dirichlet Allocation et Latent Semantic Index.\n",
    "On créé nos trois matrices d'extraction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "iteration: 1 of max_iter: 10\n",
      "iteration: 2 of max_iter: 10\n",
      "iteration: 3 of max_iter: 10\n",
      "iteration: 4 of max_iter: 10\n",
      "iteration: 5 of max_iter: 10\n",
      "iteration: 6 of max_iter: 10\n",
      "iteration: 7 of max_iter: 10\n",
      "iteration: 8 of max_iter: 10\n",
      "iteration: 9 of max_iter: 10\n",
      "iteration: 10 of max_iter: 10\n"
     ]
    }
   ],
   "source": [
    "lda_description = LatentDirichletAllocation(n_components= 10, max_iter=10, learning_method='online',verbose=True)\n",
    "data_lda_description = lda_description.fit_transform(tfidf_matrix_description)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Non-Negative Matrix Factorization Model\n",
    "nmf_description = NMF(n_components = 10)\n",
    "data_nmf_description = nmf_description.fit_transform(tfidf_matrix_description)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 258,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Latent Semantic Indexing Model using Truncated SVD\n",
    "lsi_description = TruncatedSVD(n_components = 10)\n",
    "data_lsi_description = lsi_description.fit_transform(tfidf_matrix_description)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On imprime nos divers topics pour observer s'il y a de la cohérence et si c'est exploitable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 228,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Functions for printing keywords for each topic\n",
    "def selected_topics(model, top_n = 10):\n",
    "    for idx, topic in enumerate(model.components_):\n",
    "        print(\"Topic %d:\" % (idx))\n",
    "        print([(tfidf_vectorizer_description.get_feature_names()[i], topic[i])\n",
    "                        for i in topic.argsort()[:-top_n - 1:-1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 229,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"LDA_description Model:\")\n",
    "selected_topics(lda_description)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"LSI_description Model:\")\n",
    "selected_topics(lsi_description)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 231,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NMF_description Model:\n",
      "Topic 0:\n",
      "[('développement', 0.7514516266294466), ('web', 0.6699034298620286), ('équiper', 0.6319129176587012), ('technique', 0.605646167150758), ('agile', 0.5437631600931951), ('application', 0.5317627194075365), ('javascript', 0.5309480171930088), ('coder', 0.49742467415290004), ('php', 0.46778589331301834), ('travailler', 0.4672692658843667)]\n",
      "Topic 1:\n",
      "[('team', 0.8486810110414359), ('experience', 0.7704660245704693), ('work', 0.6441642254508438), ('skills', 0.6134103282020048), ('haver', 0.574542001151327), ('new', 0.49924788674700127), ('dater', 0.4808332052680523), ('strong', 0.46294242655080375), ('business', 0.45385224724828543), ('development', 0.44998175474653296)]\n",
      "Topic 2:\n",
      "[('activus', 0.37799650959943554), ('proximité', 0.29563854462933625), ('savoir accompagner', 0.28384961842356127), ('recruter', 0.27820458871048237), ('consultant', 0.2574900176943783), ('perspective', 0.2571751815599714), ('recruter consultant', 0.25178937886923275), ('renforcer', 0.2499660625551233), ('data analytics datascience', 0.24348239280615325), ('proximité poursuivre maillage', 0.24348239280615325)]\n",
      "Topic 3:\n",
      "[('dan', 0.6695863263900329), ('client', 0.623312717622162), ('l', 0.5096564842048704), ('conseil', 0.5021778234533528), ('plaire', 0.5010443023080817), ('commercial', 0.49633966392906803), ('digital', 0.4779979340764471), ('équiper', 0.4654900246452072), ('accompagner', 0.4510489789392046), ('entreprendre', 0.4359742220817192)]\n",
      "Topic 4:\n",
      "[('cgi', 0.7757729167232194), ('information ti', 0.4596464485677409), ('ti', 0.4529454993560577), ('technologie l information', 0.35624459311517137), ('technologie l', 0.3395149746221407), ('numérique', 0.3281901419415431), ('l information ti', 0.3261381717772588), ('service technologie', 0.31903334825970425), ('l information', 0.3173205916516604), ('carrière', 0.291988256014702)]\n",
      "Topic 5:\n",
      "[('airbus', 0.5966114780155575), ('space', 0.3628157591219684), ('workforce', 0.2832789734060954), ('aircraft', 0.2663612927260494), ('diversity', 0.24857522513384345), ('application futur employment', 0.22862795850469017), ('futur employment information', 0.22862795850469017), ('futur employment', 0.22862795850469017), ('employment information', 0.22862795850469017), ('application futur', 0.22844206847077936)]\n",
      "Topic 6:\n",
      "[('info yu', 1.4298696436258647), ('info', 1.4210113325838991), ('info yu continuer', 1.3470607205313532), ('développeur', 0.6399339036009791), ('postuler signaler offrir', 0.5442039485065087), ('postuler signaler', 0.5442039485065087), ('continuer postuler signaler', 0.5442039485065087), ('rechercher', 0.5389148731080778), ('postuler', 0.5205733852419286), ('continuer postuler', 0.5183008593597755)]\n",
      "Topic 7:\n",
      "[('ingeniance', 0.6632063586941003), ('secteur banque finance', 0.3205154839083947), ('assurance ingeniance', 0.3146502922691348), ('finance assurance ingeniance', 0.3146502922691348), ('dan projet lier', 0.3144033201869291), ('technology oriented', 0.31382255444210166), ('entreprendre technology oriented', 0.31382255444210166), ('philosophie devops permettre', 0.31382255444210166), ('également entreprendre technology', 0.31382255444210166), ('leader marcher financier', 0.31382255444210166)]\n",
      "Topic 8:\n",
      "[('type d', 0.9002170523835914), ('type d emploi', 0.900153741241307), ('type', 0.8443147800770373), ('souhaité', 0.810238188364525), ('yu signaler', 0.7931810808017628), ('yu signaler offrir', 0.7931810808017628), ('souhaité yu', 0.6460574225082596), ('souhaité yu signaler', 0.5876520132215399), ('similaire', 0.5589043788311694), ('similaire souhaité', 0.517329884400731)]\n",
      "Topic 9:\n",
      "[('système', 0.5600273398814603), ('technique', 0.43322674577019216), ('donner', 0.4254606100423422), ('infrastructure', 0.41330357912964716), ('outil', 0.40150154185130144), ('dan', 0.3860952268524461), ('informatique', 0.3837221679925716), ('projet', 0.3829135948392797), ('l', 0.3817422936327766), ('d', 0.3714819975331553)]\n"
     ]
    }
   ],
   "source": [
    "print(\"NMF_description Model:\")\n",
    "selected_topics(nmf_description)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Traitement des titres avec SpaCy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ce qui a été obtenu via le traitement des descriptions n'est finalement pas satisfaisant. On essaye ici de concentrer le traitement sur les titres de chaque annonce. L'intuition c'est que l'information pour clusteriser nos annonces est plus concentrée sur les titres. Et donc la possibilité de différenciation sera plus importante."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Les étapes étant les mêmes que pour la description, on ne répétera pas les commentaires précédents. On se contentera d'analyser les résultats."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### I. Tokenize, clean, lemmatize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\David\\Miniconda3\\lib\\site-packages\\ipykernel_launcher.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "job_to_class['spacy_titre'] = [cleaning(job_to_class.loc[i, 'titre']) for i in range(len(job_to_class['titre']))]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### II. Create the TF-IDF term-documents matrix with the processed data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ici on a fait le choix de réduire les max_features à 500, et le n_gram à 1. En effet, les titres sont bien plus court, et bien plus redondants. Mais là aussi il aurait fallu tester automatiquement plusieurs valeurs et sélectionner celles qui donne les meilleures métriques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [],
   "source": [
    "#On initialise nos paramètres pour la vectorisation\n",
    "\n",
    "max_features_titres = 500\n",
    "min_n_titres = 1\n",
    "max_n_titres = 1\n",
    "\n",
    "#On crée une fonction inutile. En effet TfidfVectorizer ne peux pas gérer des documents déjà tokenisés\n",
    "\n",
    "def dummy_fun(doc): \n",
    "    return doc\n",
    "\n",
    "#On vectorise\n",
    "\n",
    "tfidf_vectorizer_titres = TfidfVectorizer( analyzer = 'word',\n",
    "                                    tokenizer = dummy_fun,\n",
    "                                    preprocessor = dummy_fun,\n",
    "                                    token_pattern = None,\n",
    "                                    max_df = 1.0, min_df = 1,\n",
    "                                    max_features = max_features_titres, sublinear_tf = True,\n",
    "                                    ngram_range=(min_n_titres, max_n_titres))\n",
    "\n",
    "\n",
    "tfidf_matrix_titres = tfidf_vectorizer_titres.fit_transform(job_to_class['spacy_titre'])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### III. Get the features !"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['',\n",
       " '7e',\n",
       " '@pwc',\n",
       " 'abap',\n",
       " 'account',\n",
       " 'achats',\n",
       " 'acquisition',\n",
       " 'actuaire',\n",
       " 'administrateur',\n",
       " 'administrator',\n",
       " 'advisory',\n",
       " 'affairer',\n",
       " 'affaires',\n",
       " 'agence',\n",
       " 'agencer',\n",
       " 'agile',\n",
       " 'al',\n",
       " 'alternance',\n",
       " 'alternant',\n",
       " 'alternante',\n",
       " 'amoa',\n",
       " 'analyse',\n",
       " 'analyst',\n",
       " 'analyste',\n",
       " 'analytics',\n",
       " 'android',\n",
       " 'anglais',\n",
       " 'angular',\n",
       " 'angularjs',\n",
       " 'api',\n",
       " 'applicatif',\n",
       " 'application',\n",
       " 'applications',\n",
       " 'apprentissage',\n",
       " 'architect',\n",
       " 'architecte',\n",
       " 'architecture',\n",
       " 'artificielle',\n",
       " 'aspnet',\n",
       " 'assistant',\n",
       " 'assistante',\n",
       " 'associate',\n",
       " 'assurance',\n",
       " 'ational',\n",
       " 'audit',\n",
       " 'auditeur',\n",
       " 'automation',\n",
       " 'automatisation',\n",
       " 'automobile',\n",
       " 'avantvente',\n",
       " 'aws',\n",
       " 'axa',\n",
       " 'azure',\n",
       " 'aéronautique',\n",
       " 'b',\n",
       " 'bancaire',\n",
       " 'banque',\n",
       " 'base',\n",
       " 'based',\n",
       " 'baser',\n",
       " 'bb',\n",
       " 'bi',\n",
       " 'big',\n",
       " 'bigdata',\n",
       " 'btob',\n",
       " 'business',\n",
       " 'bw',\n",
       " 'c',\n",
       " 'c#net',\n",
       " 'campaign',\n",
       " 'center',\n",
       " 'centre',\n",
       " 'cfo',\n",
       " 'chain',\n",
       " 'change',\n",
       " 'chantier',\n",
       " 'charge',\n",
       " 'charger',\n",
       " 'chargé',\n",
       " 'chargée',\n",
       " 'chef',\n",
       " 'chief',\n",
       " 'client',\n",
       " 'clients',\n",
       " 'clientèle',\n",
       " 'clinical',\n",
       " 'clinique',\n",
       " 'cloud',\n",
       " 'coach',\n",
       " 'cobol',\n",
       " 'commando',\n",
       " 'commerce',\n",
       " 'commercial',\n",
       " 'commerciale',\n",
       " 'communication',\n",
       " 'community',\n",
       " 'comptes',\n",
       " 'computing',\n",
       " 'concepteur',\n",
       " 'conception',\n",
       " 'conducteur',\n",
       " 'confirme',\n",
       " 'conseil',\n",
       " 'consolidation',\n",
       " 'consultant',\n",
       " 'consultante',\n",
       " 'consulter',\n",
       " 'consulting',\n",
       " 'content',\n",
       " 'contrôle',\n",
       " 'contrôleur',\n",
       " 'coordinateur',\n",
       " 'core',\n",
       " 'corporate',\n",
       " 'credit',\n",
       " 'crm',\n",
       " 'crédit',\n",
       " 'css',\n",
       " 'cto',\n",
       " 'customer',\n",
       " 'cvc',\n",
       " 'cyber',\n",
       " 'cybersécurité',\n",
       " 'd',\n",
       " 'dan',\n",
       " 'data',\n",
       " 'database',\n",
       " 'dater',\n",
       " 'dba',\n",
       " 'deep',\n",
       " 'delivery',\n",
       " 'delphi',\n",
       " 'design',\n",
       " 'designer',\n",
       " 'dev',\n",
       " 'developer',\n",
       " 'development',\n",
       " 'developpement',\n",
       " 'developper',\n",
       " 'developpeur',\n",
       " 'developpeurs',\n",
       " 'devops',\n",
       " 'digital',\n",
       " 'digitale',\n",
       " 'directeur',\n",
       " 'direction',\n",
       " 'director',\n",
       " 'django',\n",
       " 'domaine',\n",
       " 'donner',\n",
       " 'données',\n",
       " 'droit',\n",
       " 'drupal',\n",
       " 'dsi',\n",
       " 'dynamics',\n",
       " 'décisionnel',\n",
       " 'défense',\n",
       " 'déploiement',\n",
       " 'développement',\n",
       " 'développeur',\n",
       " 'développeureuse',\n",
       " 'développeurs',\n",
       " 'développeurse',\n",
       " 'développeuse',\n",
       " 'e',\n",
       " 'ecommerce',\n",
       " 'editeur',\n",
       " 'ee',\n",
       " 'embarquer',\n",
       " 'embarqué',\n",
       " 'emea',\n",
       " 'end',\n",
       " 'energie',\n",
       " 'engineer',\n",
       " 'engineering',\n",
       " 'english',\n",
       " 'enterprise',\n",
       " 'entreprendre',\n",
       " 'entreprise',\n",
       " 'environnement',\n",
       " 'epm',\n",
       " 'erp',\n",
       " 'etl',\n",
       " 'etude',\n",
       " 'etudes',\n",
       " 'europe',\n",
       " 'executive',\n",
       " 'experience',\n",
       " 'expert',\n",
       " 'exploitation',\n",
       " 'export',\n",
       " 'f',\n",
       " 'factory',\n",
       " 'field',\n",
       " 'fin',\n",
       " 'final',\n",
       " 'finance',\n",
       " 'financement',\n",
       " 'financial',\n",
       " 'financier',\n",
       " 'fintech',\n",
       " 'flight',\n",
       " 'foncier',\n",
       " 'fonctionnel',\n",
       " 'formateur',\n",
       " 'fr',\n",
       " 'france',\n",
       " 'french',\n",
       " 'frontend',\n",
       " 'fullstack',\n",
       " 'futur',\n",
       " 'gestion',\n",
       " 'gestionnaire',\n",
       " 'global',\n",
       " 'google',\n",
       " 'gouvernance',\n",
       " 'grand',\n",
       " 'grands',\n",
       " 'group',\n",
       " 'groupe',\n",
       " 'growth',\n",
       " 'génie',\n",
       " 'h',\n",
       " 'hadoop',\n",
       " 'hana',\n",
       " 'head',\n",
       " 'hip',\n",
       " 'hr',\n",
       " 'html',\n",
       " 'hybride',\n",
       " 'ia',\n",
       " 'idf',\n",
       " 'immobilier',\n",
       " 'industrie',\n",
       " 'industriel',\n",
       " 'informatica',\n",
       " 'information',\n",
       " 'informatique',\n",
       " 'infrastructure',\n",
       " 'infrastructures',\n",
       " 'ingenieur',\n",
       " 'ingénierie',\n",
       " 'innovation',\n",
       " 'integrateur',\n",
       " 'integration',\n",
       " 'intelligence',\n",
       " 'intégrateur',\n",
       " 'intégration',\n",
       " 'ios',\n",
       " 'iot',\n",
       " 'j',\n",
       " 'janvier',\n",
       " 'java',\n",
       " 'javagular',\n",
       " 'javascript',\n",
       " 'jee',\n",
       " 'js',\n",
       " 'juriste',\n",
       " 'k',\n",
       " 'kend',\n",
       " 'key',\n",
       " 'kotlin',\n",
       " 'kubernetes',\n",
       " 'l',\n",
       " 'laravel',\n",
       " 'lead',\n",
       " 'leader',\n",
       " 'learning',\n",
       " 'legal',\n",
       " 'linux',\n",
       " 'logiciel',\n",
       " 'logicielle',\n",
       " 'm',\n",
       " 'machine',\n",
       " 'magento',\n",
       " 'mainframe',\n",
       " 'maintenance',\n",
       " 'management',\n",
       " 'manager',\n",
       " 'marcher',\n",
       " 'marché',\n",
       " 'market',\n",
       " 'marketing',\n",
       " 'marketplace',\n",
       " 'media',\n",
       " 'microsoft',\n",
       " 'middleware',\n",
       " 'migration',\n",
       " 'mission',\n",
       " 'moa',\n",
       " 'mobile',\n",
       " 'modélisation',\n",
       " 'moe',\n",
       " 'months',\n",
       " 'moteur',\n",
       " 'ms',\n",
       " 'msbi',\n",
       " 'mvc',\n",
       " 'mysql',\n",
       " 'métier',\n",
       " 'n',\n",
       " 'natif',\n",
       " 'native',\n",
       " 'net',\n",
       " 'network',\n",
       " 'neuilly',\n",
       " 'neuillysurseine',\n",
       " 'niveau',\n",
       " 'nlp',\n",
       " 'node',\n",
       " 'nodejs',\n",
       " 'numérique',\n",
       " 'office',\n",
       " 'officer',\n",
       " 'offre',\n",
       " 'open',\n",
       " 'operations',\n",
       " 'ops',\n",
       " 'optimisation',\n",
       " 'opérationnel',\n",
       " 'oracle',\n",
       " 'orienter',\n",
       " 'outil',\n",
       " 'outils',\n",
       " 'owner',\n",
       " 'pacbase',\n",
       " 'partner',\n",
       " 'partners',\n",
       " 'performance',\n",
       " 'php',\n",
       " 'pilotage',\n",
       " 'pl',\n",
       " 'plateforme',\n",
       " 'platform',\n",
       " 'plm',\n",
       " 'pmo',\n",
       " 'portfolio',\n",
       " 'poste',\n",
       " 'power',\n",
       " 'pricing',\n",
       " 'principal',\n",
       " 'process',\n",
       " 'processing',\n",
       " 'product',\n",
       " 'production',\n",
       " 'produit',\n",
       " 'professional',\n",
       " 'program',\n",
       " 'programmer',\n",
       " 'programmeur',\n",
       " 'project',\n",
       " 'projet',\n",
       " 'projets',\n",
       " 'protection',\n",
       " 'préembauche',\n",
       " 'public',\n",
       " 'python',\n",
       " 'pôle',\n",
       " 'qa',\n",
       " 'qlikview',\n",
       " 'qt',\n",
       " 'quality',\n",
       " 'qualité',\n",
       " 'r',\n",
       " 'radio',\n",
       " 'rails',\n",
       " 'rd',\n",
       " 'react',\n",
       " 'reactjs',\n",
       " 'real',\n",
       " 'recette',\n",
       " 'recherche',\n",
       " 'recruiter',\n",
       " 'recrutement',\n",
       " 'ref',\n",
       " 'regional',\n",
       " 'relation',\n",
       " 'reliability',\n",
       " 'reporting',\n",
       " 'representative',\n",
       " 'research',\n",
       " 'responsable',\n",
       " 'retail',\n",
       " 'rh',\n",
       " 'risk',\n",
       " 'risquer',\n",
       " 'risques',\n",
       " 'rpa',\n",
       " 'ruby',\n",
       " 'réel',\n",
       " 'réf',\n",
       " 'référent',\n",
       " 'région',\n",
       " 'réseau',\n",
       " 'réseaux',\n",
       " 'saas',\n",
       " 'sales',\n",
       " 'salesforce',\n",
       " 'santé',\n",
       " 'sap',\n",
       " 'sas',\n",
       " 'scala',\n",
       " 'science',\n",
       " 'scientist',\n",
       " 'scrum',\n",
       " 'secteur',\n",
       " 'security',\n",
       " 'seo',\n",
       " 'server',\n",
       " 'service',\n",
       " 'services',\n",
       " 'sharepoint',\n",
       " 'sig',\n",
       " 'simulation',\n",
       " 'site',\n",
       " 'smart',\n",
       " 'social',\n",
       " 'software',\n",
       " 'solution',\n",
       " 'solutions',\n",
       " 'spark',\n",
       " 'specialist',\n",
       " 'spring',\n",
       " 'spécialiste',\n",
       " 'sql',\n",
       " 'sr',\n",
       " 'sre',\n",
       " 'st',\n",
       " 'stack',\n",
       " 'start',\n",
       " 'startup',\n",
       " 'statistique',\n",
       " 'strategic',\n",
       " 'strategy',\n",
       " 'stratégie',\n",
       " 'success',\n",
       " 'supply',\n",
       " 'support',\n",
       " 'symfony',\n",
       " 'system',\n",
       " 'systeme',\n",
       " 'systems',\n",
       " 'système',\n",
       " 'systèmes',\n",
       " 'sécurité',\n",
       " 'sédentaire',\n",
       " 'sénior',\n",
       " 'sûreté',\n",
       " 'tableau',\n",
       " 'talend',\n",
       " 'talent',\n",
       " 'team',\n",
       " 'tech',\n",
       " 'technical',\n",
       " 'technicien',\n",
       " 'technico',\n",
       " 'technique',\n",
       " 'technologie',\n",
       " 'technologies',\n",
       " 'technology',\n",
       " 'telecom',\n",
       " 'temps',\n",
       " 'test',\n",
       " 'testeur',\n",
       " 'tests',\n",
       " 'traffic',\n",
       " 'trafic',\n",
       " 'traitement',\n",
       " 'transformation',\n",
       " 'transport',\n",
       " 'travail',\n",
       " 'télécom',\n",
       " 'ui',\n",
       " 'une',\n",
       " 'unity',\n",
       " 'unix',\n",
       " 'urgent',\n",
       " 'user',\n",
       " 'ux',\n",
       " 'validation',\n",
       " 'vba',\n",
       " 'vente',\n",
       " 'virtualisation',\n",
       " 'visualisation',\n",
       " 'vue',\n",
       " 'vuejs',\n",
       " 'web',\n",
       " 'webdev',\n",
       " 'webmarketing',\n",
       " 'windev',\n",
       " 'windows',\n",
       " 'wpf',\n",
       " 'x',\n",
       " 'xp',\n",
       " '«',\n",
       " '»',\n",
       " 'éditeur',\n",
       " 'équiper',\n",
       " 'étude',\n",
       " '–']"
      ]
     },
     "execution_count": 213,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features_titres = tfidf_vectorizer_titres.get_feature_names()\n",
    "features_titres"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### IV. Find the (features x topics) matric with the LDA/LSI/NMF and generate groups"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "iteration: 1 of max_iter: 10\n",
      "iteration: 2 of max_iter: 10\n",
      "iteration: 3 of max_iter: 10\n",
      "iteration: 4 of max_iter: 10\n",
      "iteration: 5 of max_iter: 10\n",
      "iteration: 6 of max_iter: 10\n",
      "iteration: 7 of max_iter: 10\n",
      "iteration: 8 of max_iter: 10\n",
      "iteration: 9 of max_iter: 10\n",
      "iteration: 10 of max_iter: 10\n"
     ]
    }
   ],
   "source": [
    "lda_titres = LatentDirichletAllocation(n_components= 20, max_iter=10, learning_method='online',verbose=True)\n",
    "data_lda_titres = lda_titres.fit_transform(tfidf_matrix_titres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_topic_20 = pd.DataFrame(data_lda_titres)\n",
    "df_topic_20\n",
    "df_topic_20.to_csv(\"topic_20.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Non-Negative Matrix Factorization Model\n",
    "nmf_titres = NMF(n_components = 20)\n",
    "data_nmf_titres = nmf_titres.fit_transform(tfidf_matrix_titres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Latent Semantic Indexing Model using Truncated SVD\n",
    "lsi_titres = TruncatedSVD(n_components = 15)\n",
    "data_lsi_titres = lsi_titres.fit_transform(tfidf_matrix_titres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Functions for printing keywords for each topic\n",
    "def selected_topics(model, top_n = 15):\n",
    "    for idx, topic in enumerate(model.components_):\n",
    "        print(\"Topic %d:\" % (idx))\n",
    "        print([(tfidf_vectorizer_titres.get_feature_names()[i], topic[i])\n",
    "                        for i in topic.argsort()[:-top_n - 1:-1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(\"LDA_titres Model:\")\n",
    "# type(selected_topics(lda_titres))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(\"LSI_titres Model:\")\n",
    "# selected_topics(lsi_titres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NMF_titres Model:\n",
      "Topic 0:\n",
      "[('développeur', 6.607743426447681), ('python', 0.47146232538874716), ('informatique', 0.27540543664176687), ('android', 0.24001887597081448), ('concepteur', 0.21767208377440908), ('angular', 0.17951855578926051), ('logiciel', 0.15387793500966132), ('react', 0.15319435458367406), ('ios', 0.1502756565441051), ('mobile', 0.14563296980997023), ('analyste', 0.13976105961376137), ('javascript', 0.12480775094211563), ('sql', 0.12240896804006231), ('kend', 0.11457165560364041), ('salesforce', 0.10446063022250011)]\n",
      "Topic 1:\n",
      "[('data', 5.026926171198431), ('scientist', 2.3823911269767124), ('big', 1.1641682131497664), ('science', 0.2126789084393263), ('lead', 0.15741633942180944), ('architect', 0.13626924367843748), ('–', 0.12403698528564958), ('analyste', 0.10325436444980121), ('analytics', 0.09859649758107236), ('tech', 0.08167653495857265), ('ingenieur', 0.06105126962463851), ('consultante', 0.060809218191784795), ('protection', 0.04603180272909367), ('intelligence', 0.04519270289011762), ('management', 0.03610497061186018)]\n",
      "Topic 2:\n",
      "[('devops', 4.758734407728827), ('cloud', 0.3408094125314966), ('système', 0.23774500617102592), ('intégrateur', 0.11325202499008911), ('aws', 0.10620695213561779), ('expert', 0.08483973520542026), ('linux', 0.08425828843570993), ('ingenieur', 0.08386125173640221), ('production', 0.08286757096821509), ('azure', 0.07565117363187163), ('administrateur', 0.06171750651675794), ('sre', 0.05578463417133072), ('–', 0.05528015643337096), ('intégration', 0.05519807502485477), ('python', 0.05376835739694362)]\n",
      "Topic 3:\n",
      "[('consultant', 5.131540755089077), ('bi', 0.5598008493793545), ('sap', 0.27244647308969033), ('fonctionnel', 0.20162476403746524), ('cloud', 0.1912896525050736), ('intelligence', 0.19026114108069586), ('amoa', 0.16229599639688025), ('talend', 0.15555656266936171), ('–', 0.1460325044098184), ('moa', 0.14424974467030696), ('finance', 0.1319462969013693), ('big', 0.12816097960680137), ('technique', 0.12003953508338126), ('décisionnel', 0.1124710366108691), ('microsoft', 0.10675610661066928)]\n",
      "Topic 4:\n",
      "[('java', 4.425296034199256), ('jee', 0.8050489283373847), ('j', 0.609437065887714), ('lead', 0.23636789262585187), ('angular', 0.16069162909784987), ('ee', 0.12916441606167217), ('spring', 0.10268684016153384), ('technique', 0.09043700260649425), ('tech', 0.08272915496703886), ('expert', 0.07447199825993914), ('k', 0.07194180089152562), ('leader', 0.06522509259656768), ('kend', 0.06378072582011282), ('développeur', 0.05150798776016324), ('concepteur', 0.04479435320952565)]\n",
      "Topic 5:\n",
      "[('projet', 3.160776260196927), ('chef', 3.0804854788457923), ('technique', 0.4175334221453985), ('digital', 0.34623426909798066), ('informatique', 0.2144269377167403), ('assistant', 0.13742189423854603), ('fonctionnel', 0.1246769504944254), ('moa', 0.12314460130513445), ('infrastructure', 0.12127649778035107), ('marketing', 0.1173784256325122), ('bi', 0.11211551930053432), ('sap', 0.10930452080788441), ('–', 0.09123559128002427), ('projets', 0.08209351508578999), ('amoa', 0.08199127235149407)]\n",
      "Topic 6:\n",
      "[('manager', 4.281809808439952), ('product', 0.9240356432506971), ('project', 0.47987039497569484), ('account', 0.4579115209360199), ('marketing', 0.2750958157834), ('f', 0.22492112050342022), ('m', 0.21825137341645615), ('france', 0.21107831908203104), ('owner', 0.19582909878986815), ('sales', 0.18976677373339357), ('digital', 0.1394522342565057), ('data', 0.12700284777317564), ('–', 0.12580756829759382), ('customer', 0.12262175575279222), ('success', 0.11625222882291603)]\n",
      "Topic 7:\n",
      "[('analyst', 4.354732397962548), ('business', 0.8459166729378917), ('data', 0.6220922611576896), ('hip', 0.17722625375464254), ('marketing', 0.17552446360564414), ('digital', 0.16545948422197834), ('operations', 0.1136739214246026), ('alternance', 0.10178332955345525), ('m', 0.09183979543422026), ('f', 0.08475166233818969), ('assistant', 0.06587810516340743), ('crm', 0.06540531593748454), ('finance', 0.06522358645905689), ('research', 0.06495798246904981), ('dater', 0.0584525619004241)]\n",
      "Topic 8:\n",
      "[('web', 7.370557265836519), ('mobile', 0.5159480040736233), ('intégrateur', 0.23044766140223327), ('application', 0.20806589589061844), ('alternance', 0.20399760657287916), ('–', 0.18458995286998311), ('analyste', 0.14549910144289996), ('développeurse', 0.1128024284012125), ('analytics', 0.09210718247528032), ('services', 0.08247919815134666), ('développeureuse', 0.08159647519180287), ('sig', 0.06861600694757265), ('designer', 0.06294349843807862), ('rd', 0.05342687850992736), ('service', 0.04177740195299749)]\n",
      "Topic 9:\n",
      "[('c', 6.693187554425138), ('qt', 0.28963491970013716), ('embarquer', 0.1582416245653884), ('finance', 0.15784466116641166), ('aspnet', 0.10550290678535552), ('ingenieur', 0.09461301639420498), ('sql', 0.09169774649452266), ('linux', 0.08739741010938273), ('–', 0.07846706824209808), ('windows', 0.07798980947086506), ('développeur', 0.0638806721629565), ('mvc', 0.0606545183488338), ('logiciel', 0.05096048988087943), ('embarqué', 0.04919921520655706), ('software', 0.04621159239427777)]\n",
      "Topic 10:\n",
      "[('engineer', 4.280086119260653), ('software', 1.0279717861282043), ('data', 0.5559815811632561), ('m', 0.25965970652485704), ('f', 0.25955112650775214), ('cloud', 0.1559162303072584), ('reliability', 0.14606472049832023), ('kend', 0.14451646134497084), ('site', 0.14050045764268135), ('–', 0.11257072489075044), ('python', 0.11184103704440129), ('qa', 0.08863480345886442), ('rd', 0.08474191968707757), ('hip', 0.08287314989412745), ('sales', 0.08121171994661518)]\n",
      "Topic 11:\n",
      "[('end', 4.151184361721279), ('k', 2.0477252635115515), ('developer', 0.20648484858741634), ('react', 0.17155441106345462), ('développeurse', 0.1541340270901685), ('nodejs', 0.12379239659892453), ('lead', 0.1227743045719942), ('javascript', 0.11189865458160538), ('reactjs', 0.1056871575747038), ('js', 0.07309307072258724), ('angular', 0.07201215660678083), ('expert', 0.06992426890336884), ('developpeurs', 0.04686748109178535), ('développeuse', 0.04635954141056069), ('offre', 0.037928468340753176)]\n",
      "Topic 12:\n",
      "[('architecte', 4.319639413629371), ('cloud', 1.1091216066030076), ('technique', 0.8639059497646837), ('logiciel', 0.36954740832215516), ('informatique', 0.31949747768756026), ('sécurité', 0.2728297304588277), ('big', 0.26148185803921065), ('azure', 0.24461631729863315), ('infrastructure', 0.24280070358675818), ('système', 0.1984769460350377), ('expert', 0.1963420368642041), ('solution', 0.18412539174490847), ('réseau', 0.1765128395178004), ('applicatif', 0.16290417431122134), ('–', 0.13315150848547255)]\n",
      "Topic 13:\n",
      "[('php', 4.69482560498736), ('symfony', 2.410392833345353), ('lead', 0.29683301943923207), ('développeurs', 0.18408595229090383), ('drupal', 0.11997399835680218), ('laravel', 0.11286701951632128), ('mysql', 0.10800162636358333), ('développeur', 0.09889823327243288), ('kend', 0.09710538827870252), ('k', 0.08759538382096714), ('magento', 0.08184423781904443), ('js', 0.07650100540292416), ('developpeurs', 0.0745305938356917), ('développeurse', 0.0633359313121134), ('dev', 0.052152933092682256)]\n",
      "Topic 14:\n",
      "[('developer', 3.219908406194176), ('business', 2.960591468949619), ('lead', 0.47399567885585453), ('intelligence', 0.4212539053018632), ('commercial', 0.16188083604320802), ('development', 0.12564320347981267), ('btob', 0.08207064750730962), ('python', 0.08036029018799476), ('kend', 0.07878830951606004), ('–', 0.0626472343246635), ('tech', 0.05256343058496674), ('bb', 0.051107751749789584), ('software', 0.04845213081878234), ('alternance', 0.048446428006584566), ('start', 0.041729533253070564)]\n",
      "Topic 15:\n",
      "[('stack', 3.995548363233217), ('js', 0.1698546518414546), ('javascript', 0.1566781062961445), ('angular', 0.1448956168268987), ('–', 0.139837997905136), ('lead', 0.12812956342602175), ('developer', 0.10359985546323795), ('développeurse', 0.09586322338449929), ('software', 0.09534566782834426), ('node', 0.06763616720489739), ('react', 0.06434572947637864), ('tech', 0.058672799535272574), ('nodejs', 0.047380231109096206), ('développeurs', 0.04737883414048349), ('dev', 0.0429168362585707)]\n",
      "Topic 16:\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('développement', 3.7099226173944455), ('d', 1.350199542920031), ('responsable', 1.0690385600441057), ('étude', 0.9857774863932953), ('logiciel', 0.6131697293941284), ('informatique', 0.4404888456345535), ('python', 0.3935454030128623), ('etudes', 0.36127866728636915), ('commercial', 0.3393525292114552), ('chargé', 0.28519828353064225), ('–', 0.2304376219048842), ('système', 0.19734713620808073), ('application', 0.1953694588861208), ('etude', 0.18318932244944464), ('affairer', 0.18006398646235325)]\n",
      "Topic 17:\n",
      "[('net', 3.9408727798697147), ('lead', 0.4177167829975332), ('core', 0.17441436571952815), ('tech', 0.12095814845653903), ('azure', 0.1153934401930175), ('angular', 0.08064404323560315), ('développeurs', 0.06275732817945033), ('wpf', 0.04792021153091603), ('développeur', 0.041443496158863995), ('expert', 0.040347671817452746), ('editeur', 0.039430616759255833), ('sharepoint', 0.033873749429326416), ('developer', 0.03216938551850057), ('technique', 0.03149021349635194), ('sénior', 0.02798541473661184)]\n",
      "Topic 18:\n",
      "[('fullstack', 4.4279427332400845), ('angular', 0.555167023467516), ('js', 0.41805520655914563), ('javascript', 0.33789076733600837), ('react', 0.2131983758983963), ('nodejs', 0.205154812714365), ('developer', 0.18724190353432604), ('assistant', 0.09596225931130319), ('développeurse', 0.09404560210078962), ('node', 0.07859801265753047), ('reactjs', 0.0762380850531405), ('javagular', 0.05956664476228081), ('vuejs', 0.05829876072701527), ('développeurs', 0.040839530275799216), ('final', 0.039875409919486654)]\n",
      "Topic 19:\n",
      "[('developpeur', 3.660016125417995), ('informatique', 0.23241665413926071), ('python', 0.2112299603767388), ('lead', 0.1955658254265532), ('analyste', 0.16959511422390858), ('concepteur', 0.15105307240372084), ('android', 0.11440048788322803), ('angular', 0.11019798927801022), ('js', 0.08016411540235859), ('sql', 0.07699874568886123), ('mobile', 0.07664264719465447), ('ios', 0.07590916717182196), ('salesforce', 0.07381107963308116), ('ingenieur', 0.059791154901132244), ('logiciel', 0.05946323998308201)]\n",
      "<class 'NoneType'>\n"
     ]
    }
   ],
   "source": [
    "print(\"NMF_titres Model:\")\n",
    "print(type(selected_topics(nmf_titres)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_topic_20 = pd.DataFrame(data_nmf_titres)\n",
    "df_topic_20.to_csv(\"topic_20.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "      <th>13</th>\n",
       "      <th>14</th>\n",
       "      <th>15</th>\n",
       "      <th>16</th>\n",
       "      <th>17</th>\n",
       "      <th>18</th>\n",
       "      <th>19</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>count</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "      <td>19418.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>mean</td>\n",
       "      <td>0.014234</td>\n",
       "      <td>0.008495</td>\n",
       "      <td>0.007349</td>\n",
       "      <td>0.007135</td>\n",
       "      <td>0.008022</td>\n",
       "      <td>0.006779</td>\n",
       "      <td>0.008153</td>\n",
       "      <td>0.005167</td>\n",
       "      <td>0.003239</td>\n",
       "      <td>0.002750</td>\n",
       "      <td>0.005923</td>\n",
       "      <td>0.004774</td>\n",
       "      <td>0.005947</td>\n",
       "      <td>0.004171</td>\n",
       "      <td>0.006431</td>\n",
       "      <td>0.004675</td>\n",
       "      <td>0.007009</td>\n",
       "      <td>0.004796</td>\n",
       "      <td>0.004365</td>\n",
       "      <td>0.006066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>std</td>\n",
       "      <td>0.027063</td>\n",
       "      <td>0.026370</td>\n",
       "      <td>0.032176</td>\n",
       "      <td>0.025587</td>\n",
       "      <td>0.030198</td>\n",
       "      <td>0.029135</td>\n",
       "      <td>0.028388</td>\n",
       "      <td>0.025032</td>\n",
       "      <td>0.015333</td>\n",
       "      <td>0.016313</td>\n",
       "      <td>0.026082</td>\n",
       "      <td>0.024399</td>\n",
       "      <td>0.023883</td>\n",
       "      <td>0.021561</td>\n",
       "      <td>0.026627</td>\n",
       "      <td>0.025976</td>\n",
       "      <td>0.021943</td>\n",
       "      <td>0.026361</td>\n",
       "      <td>0.022411</td>\n",
       "      <td>0.028576</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>min</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25%</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>50%</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>75%</td>\n",
       "      <td>0.026501</td>\n",
       "      <td>0.000168</td>\n",
       "      <td>0.000163</td>\n",
       "      <td>0.000937</td>\n",
       "      <td>0.000048</td>\n",
       "      <td>0.000794</td>\n",
       "      <td>0.000887</td>\n",
       "      <td>0.000455</td>\n",
       "      <td>0.000047</td>\n",
       "      <td>0.000014</td>\n",
       "      <td>0.000583</td>\n",
       "      <td>0.000084</td>\n",
       "      <td>0.000519</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000262</td>\n",
       "      <td>0.000010</td>\n",
       "      <td>0.002600</td>\n",
       "      <td>0.000059</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000381</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>max</td>\n",
       "      <td>0.148911</td>\n",
       "      <td>0.164515</td>\n",
       "      <td>0.207756</td>\n",
       "      <td>0.188958</td>\n",
       "      <td>0.213124</td>\n",
       "      <td>0.220277</td>\n",
       "      <td>0.211917</td>\n",
       "      <td>0.214926</td>\n",
       "      <td>0.134359</td>\n",
       "      <td>0.148693</td>\n",
       "      <td>0.213726</td>\n",
       "      <td>0.205111</td>\n",
       "      <td>0.201712</td>\n",
       "      <td>0.173640</td>\n",
       "      <td>0.222476</td>\n",
       "      <td>0.247675</td>\n",
       "      <td>0.192563</td>\n",
       "      <td>0.249517</td>\n",
       "      <td>0.217236</td>\n",
       "      <td>0.267637</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 0             1             2             3             4   \\\n",
       "count  19418.000000  19418.000000  19418.000000  19418.000000  19418.000000   \n",
       "mean       0.014234      0.008495      0.007349      0.007135      0.008022   \n",
       "std        0.027063      0.026370      0.032176      0.025587      0.030198   \n",
       "min        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "25%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "50%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "75%        0.026501      0.000168      0.000163      0.000937      0.000048   \n",
       "max        0.148911      0.164515      0.207756      0.188958      0.213124   \n",
       "\n",
       "                 5             6             7             8             9   \\\n",
       "count  19418.000000  19418.000000  19418.000000  19418.000000  19418.000000   \n",
       "mean       0.006779      0.008153      0.005167      0.003239      0.002750   \n",
       "std        0.029135      0.028388      0.025032      0.015333      0.016313   \n",
       "min        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "25%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "50%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "75%        0.000794      0.000887      0.000455      0.000047      0.000014   \n",
       "max        0.220277      0.211917      0.214926      0.134359      0.148693   \n",
       "\n",
       "                 10            11            12            13            14  \\\n",
       "count  19418.000000  19418.000000  19418.000000  19418.000000  19418.000000   \n",
       "mean       0.005923      0.004774      0.005947      0.004171      0.006431   \n",
       "std        0.026082      0.024399      0.023883      0.021561      0.026627   \n",
       "min        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "25%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "50%        0.000000      0.000000      0.000000      0.000000      0.000000   \n",
       "75%        0.000583      0.000084      0.000519      0.000000      0.000262   \n",
       "max        0.213726      0.205111      0.201712      0.173640      0.222476   \n",
       "\n",
       "                 15            16            17            18            19  \n",
       "count  19418.000000  19418.000000  19418.000000  19418.000000  19418.000000  \n",
       "mean       0.004675      0.007009      0.004796      0.004365      0.006066  \n",
       "std        0.025976      0.021943      0.026361      0.022411      0.028576  \n",
       "min        0.000000      0.000000      0.000000      0.000000      0.000000  \n",
       "25%        0.000000      0.000000      0.000000      0.000000      0.000000  \n",
       "50%        0.000000      0.000000      0.000000      0.000000      0.000000  \n",
       "75%        0.000010      0.002600      0.000059      0.000000      0.000381  \n",
       "max        0.247675      0.192563      0.249517      0.217236      0.267637  "
      ]
     },
     "execution_count": 233,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_topic_20.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "L'application de notre méthode de NLP sur les titres s'avère donner des extractions de topic plus cohérent que celle sur les descriptions. Toutefois l'ajout de notre matrice NMF dans les features de notre dataset nous fait gagner une accuracy relativement marginale. Et malheureusement les topics ne sont pas suffisamment cohérent pour alimenter la data analysis côté client.\n",
    "Peut-être que la libraire Bert pourrait s'avérer plus utile sur ces deux niveaux."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
