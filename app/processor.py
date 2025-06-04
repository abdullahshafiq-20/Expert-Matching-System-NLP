import re
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter
import numpy as np
from datetime import datetime

# Try to import NLTK and spaCy, but fall back gracefully
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVProcessor:
    def __init__(self):
        self.nlp = None
        self.stop_words = set()
        
        # Initialize NLP libraries if available
        self._init_nltk()
        self._init_spacy()
        
        # Enhanced skill patterns for different domains
        self.skill_patterns = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c\\+\\+', 'c#', 'php', 'matlab', 
                'r\\b', 'sql', 'scala', 'kotlin', 'swift', 'typescript', 'ruby',
                'go', 'rust', 'perl', 'dart', 'objective-c', 'assembly', 'fortran',
                'cobol', 'pascal', 'basic', 'lisp', 'prolog', 'haskell', 'erlang',
                'clojure', 'f#', 'julia', 'lua', 'shell', 'bash', 'powershell'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 
                'flask', 'asp\\.net', 'spring', 'laravel', 'express', 'bootstrap',
                'jquery', 'webpack', 'babel', 'sass', 'less', 'next\\.js', 'nuxt\\.js',
                'gatsby', 'graphql', 'rest api', 'soap', 'websocket', 'webgl',
                'three\\.js', 'd3\\.js', 'socket\\.io', 'redux', 'mobx', 'vuex',
                'tailwind', 'material-ui', 'ant design', 'chakra ui'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'sql server',
                'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'neo4j', 'couchdb',
                'firebase', 'firestore', 'mariadb', 'hbase', 'couchbase', 'influxdb',
                'timescaledb', 'cockroachdb', 'arangodb', 'rethinkdb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'google cloud', 'gcp', 'heroku', 'digitalocean',
                'kubernetes', 'docker', 'openshift', 'cloudflare', 'linode', 'vultr',
                'alibaba cloud', 'oracle cloud', 'ibm cloud', 'rackspace', 'terraform',
                'ansible', 'puppet', 'chef', 'jenkins', 'gitlab ci', 'github actions',
                'circleci', 'travis ci', 'teamcity', 'bamboo'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data science', 'ai', 
                'artificial intelligence', 'tensorflow', 'pytorch', 'scikit-learn',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi',
                'jupyter', 'kaggle', 'statistics', 'regression', 'classification',
                'clustering', 'nlp', 'computer vision', 'opencv', 'spark', 'hadoop',
                'hive', 'pig', 'impala', 'kafka', 'storm', 'flink', 'airflow',
                'mlflow', 'kubeflow', 'fastai', 'keras', 'theano', 'caffe',
                'mxnet', 'xgboost', 'lightgbm', 'catboost'
            ],
            'development_tools': [
                'git', 'github', 'gitlab', 'bitbucket', 'jenkins', 'docker',
                'kubernetes', 'terraform', 'ansible', 'vagrant', 'maven',
                'gradle', 'npm', 'yarn', 'pip', 'conda', 'composer', 'cargo',
                'nuget', 'gem', 'bower', 'webpack', 'rollup', 'parcel', 'vite',
                'eslint', 'prettier', 'sonarqube', 'jira', 'confluence', 'trello',
                'asana', 'slack', 'teams', 'zoom', 'postman', 'insomnia',
                'swagger', 'openapi', 'graphql playground'
            ],
            'methodologies': [
                'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd',
                'microservices', 'api design', 'restful', 'graphql', 'soap',
                'waterfall', 'spiral', 'rad', 'prototype', 'lean', 'six sigma',
                'itil', 'cmmi', 'pmp', 'prince2', 'safe', 'crystal', 'xp',
                'feature driven development', 'domain driven design'
            ],
            'research_skills': [
                'research', 'analysis', 'peer review', 'technical writing',
                'literature review', 'data analysis', 'statistical analysis',
                'experimental design', 'survey design', 'qualitative research',
                'quantitative research', 'mixed methods', 'case study',
                'ethnography', 'grounded theory', 'action research',
                'systematic review', 'meta-analysis', 'content analysis',
                'discourse analysis', 'thematic analysis', 'coding',
                'research ethics', 'grant writing', 'proposal writing',
                'academic writing', 'publication', 'citation management',
                'research methodology', 'research design'
            ],
            'teaching_skills': [
                'teaching', 'lecturing', 'course design', 'curriculum development',
                'lesson planning', 'assessment design', 'student evaluation',
                'classroom management', 'online teaching', 'blended learning',
                'instructional design', 'pedagogy', 'andragogy', 'mentoring',
                'tutoring', 'academic advising', 'student supervision',
                'thesis supervision', 'dissertation supervision', 'lab instruction',
                'workshop facilitation', 'training', 'professional development',
                'educational technology', 'learning management systems',
                'moodle', 'blackboard', 'canvas', 'google classroom'
            ],
            'soft_skills': [
                'leadership', 'teamwork', 'communication', 'problem solving',
                'critical thinking', 'time management', 'project management',
                'conflict resolution', 'negotiation', 'presentation',
                'public speaking', 'interpersonal skills', 'emotional intelligence',
                'adaptability', 'creativity', 'innovation', 'analytical thinking',
                'decision making', 'strategic thinking', 'collaboration',
                'mentoring', 'coaching', 'facilitation', 'networking',
                'cultural awareness', 'diversity and inclusion'
            ]
        }
        
        # Enhanced blacklist of terms that are not skills
        self.skill_blacklist = {
            # Universities and institutions
            'air university', 'comsats', 'comsats institute', 'institute of information technology',
            'university', 'college', 'institute', 'department', 'school', 'academy',
            'education', 'islamabad', 'lahore', 'karachi', 'peshawar', 'quetta',
            'bahawalpur', 'faisalabad', 'multan', 'rawalpindi', 'gujranwala',
            
            # Companies and organizations
            'daewoo', 'microsoft', 'google', 'facebook', 'apple', 'amazon',
            'company', 'corporation', 'ltd', 'limited', 'inc', 'incorporated',
            'organization', 'firm', 'enterprise', 'business', 'group',
            
            # Generic terms
            'analysis', 'research', 'skills', 'achievements', 'achievement',
            'experience', 'work', 'project', 'projects', 'development',
            'management', 'system', 'systems', 'technology', 'technologies',
            'computer', 'science', 'sciences', 'engineering', 'program',
            'programming', 'software', 'hardware', 'technical', 'digital',
            
            # Document formatting and headers
            'page', 'references', 'contact', 'information', 'address',
            'phone', 'email', 'cv', 'resume', 'curriculum vitae',
            'objective', 'summary', 'profile', 'personal', 'details',
            'education', 'qualifications', 'certifications', 'awards',
            'publications', 'interests', 'hobbies', 'languages',
            
            # Degree names and levels
            'bachelor', 'bachelors', 'master', 'masters', 'phd', 'doctorate',
            'diploma', 'certificate', 'degree', 'graduation', 'undergraduate',
            'graduate', 'postgraduate', 'bsc', 'msc', 'btech', 'mtech',
            
            # Specific noise terms
            '3d brain', 'bad news bearer', 'congress', 'snooker club',
            'management system', 'using', 'suit', 'office suit', 'ms office',
            'acceptance', 'lecturer', 'hafiz', 'page 1', 'page 2',
            '-----------------------', 'current', 'previous', 'present',
            'past', 'future', 'ongoing', 'completed', 'in progress'
        }
        
        # Enhanced education level keywords
        self.education_patterns = {
            'phd': [
                'phd', 'ph\\.d', 'doctorate', 'doctoral', 'd\\.phil', 'dphil',
                'doctor of philosophy', 'doctor of science', 'd\\.sc', 'dsc'
            ],
            'masters': [
                'masters?', 'm\\.s', 'm\\.sc', 'msc', 'm\\.a', 'ma', 'mba', 
                'm\\.tech', 'mtech', 'm\\.e', 'me', 'm\\.eng', 'meng',
                'master of science', 'master of arts', 'master of engineering',
                'master of technology', 'master of business administration'
            ],
            'bachelors': [
                'bachelors?', 'b\\.s', 'b\\.sc', 'bsc', 'b\\.a', 'ba', 
                'b\\.tech', 'btech', 'b\\.e', 'be', 'b\\.eng', 'beng',
                'bachelor of science', 'bachelor of arts', 'bachelor of engineering',
                'bachelor of technology', 'bachelor of computer science',
                'bachelor of information technology'
            ],
            'diploma': [
                'diploma', 'certificate', 'certification', 'associate degree',
                'a\\.s', 'as', 'a\\.a', 'aa', 'associate of science',
                'associate of arts', 'associate of applied science'
            ]
        }
        
        # Enhanced experience extraction patterns
        self.experience_patterns = [
            r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp).*?(\d+)\s*(?:\+)?\s*(?:years?|yrs?)',
            r'(\d{4})\s*-\s*(?:present|current|\d{4})',
            r'(?:from|since)\s*(\d{4})',
            r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:in|with|of)',
            r'(?:worked|working|served|serving)\s*(?:for|as|in)\s*(\d+)\s*(?:years?|yrs?)',
            r'(?:total|overall)\s*(?:of\s*)?(\d+)\s*(?:years?|yrs?)\s*(?:experience|exp)',
            r'(?:experience|exp)\s*(?:of\s*)?(\d+)\s*(?:\+)?\s*(?:years?|yrs?)',
            r'(?:minimum|at least)\s*(\d+)\s*(?:years?|yrs?)\s*(?:experience|exp)'
        ]
        
        # Date patterns for experience extraction
        self.date_patterns = [
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*-\s*(?:present|current|now|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}\s*-\s*(?:present|current|now|\d{1,2}/\d{1,2}/\d{4})',
            r'\d{1,2}-\d{1,2}-\d{4}\s*-\s*(?:present|current|now|\d{1,2}-\d{1,2}-\d{4})',
            r'\d{4}\s*-\s*(?:present|current|now|\d{4})'
        ]
        
        # Contact information patterns
        self.contact_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': [
                r'\b(?:\+92|0092|92)?\s*\d{3}\s*\d{7}\b',  # Pakistani format
                r'\b\d{3}-\d{3}-\d{4}\b',  # US format
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # US format with parentheses
                r'\b\d{10,11}\b'  # Simple 10-11 digit number
            ],
            'location': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*([A-Z]{2})\b'  # City, State format
            ],
            'linkedin': r'(?:linkedin|linked-in|linked in)[\s:]*[\w\-\.]+',
            'github': r'(?:github|git-hub|git hub)[\s:]*[\w\-\.]+',
            'website': r'(?:www\.|https?://)?[\w\-\.]+\.[a-z]{2,}(?:/[\w\-\.]+)*'
        }
        
        # Certification patterns
        self.certification_patterns = [
            'certified', 'certification', 'certificate', 'license', 'credential',
            'aws certified', 'microsoft certified', 'google certified', 'cisco certified',
            'pmp', 'scrum master', 'agile', 'itil', 'prince2', 'comptia',
            'oracle certified', 'red hat certified', 'vmware certified',
            'salesforce certified', 'ibm certified', 'apple certified',
            'adobe certified', 'autodesk certified', 'cisco ccna', 'cisco ccnp',
            'cisco ccie', 'microsoft mcp', 'microsoft mcsd', 'microsoft mcse',
            'microsoft mct', 'microsoft mta', 'microsoft mcitp', 'microsoft mcts',
            'microsoft mcad', 'microsoft mcsa', 'microsoft mcdba', 'microsoft mcdst',
            'microsoft mcm', 'microsoft mca', 'microsoft mcpd', 'microsoft mcts',
            'microsoft mcitp', 'microsoft mcad', 'microsoft mcsa', 'microsoft mcdba',
            'microsoft mcdst', 'microsoft mcm', 'microsoft mca', 'microsoft mcpd'
        ]
    
    def _init_nltk(self):
        """Initialize NLTK with required data"""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available. Basic text processing will be used.")
            return
        
        try:
            # Download required NLTK data
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        logger.info("NLTK initialized successfully")
    
    def _init_spacy(self):
        """Initialize spaCy model"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Advanced NLP features disabled.")
            return
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except IOError:
            logger.warning("spaCy English model not found. Please install it: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove page numbers and headers
        text = re.sub(r'page\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-{3,}', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,;:()\-+/]', ' ', text)
        
        return text.strip()

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills using regex patterns and NLP"""
        text_lower = self._clean_text(text.lower())
        extracted_skills = {category: set() for category in self.skill_patterns.keys()}
        
        # Extract skills using regex patterns
        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.findall(r'\b' + pattern + r'\b', text_lower, re.IGNORECASE)
                if matches:
                    # Clean and normalize matches
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1]
                        clean_skill = self._clean_skill_name(match)
                        if clean_skill and len(clean_skill) > 1 and self._is_valid_skill(clean_skill):
                            extracted_skills[category].add(clean_skill)
        
        # Use spaCy for named entity recognition if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PRODUCT', 'ORG']:
                        skill = self._clean_skill_name(ent.text.lower())
                        if skill and len(skill) > 1 and self._is_valid_skill(skill):
                            if self._is_technology_term(skill):
                                # Try to categorize the skill
                                categorized = False
                                for category, patterns in self.skill_patterns.items():
                                    if any(pattern in skill.lower() for pattern in patterns):
                                        extracted_skills[category].add(skill)
                                        categorized = True
                                        break
                                if not categorized:
                                    extracted_skills['development_tools'].add(skill)
            except Exception as e:
                logger.warning(f"spaCy processing failed: {e}")
        
        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in extracted_skills.items()}

    def extract_experience_years(self, text: str) -> Dict[str, int]:
        """Extract years of experience from text"""
        years = set()
        experience_info = {
            'total_years': 0,
            'relevant_years': 0,
            'teaching_years': 0,
            'research_years': 0
        }
        
        # Extract years using patterns
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        year = int(match[0]) if match[0] else int(match[1])
                    else:
                        year = int(match)
                    
                    # Validate reasonable experience range
                    if 0 <= year <= 50:
                        years.add(year)
                except (ValueError, IndexError):
                    continue
        
        # Extract date ranges
        for pattern in self.date_patterns:
            date_ranges = re.findall(pattern, text, re.IGNORECASE)
            for date_range in date_ranges:
                try:
                    if 'present' in date_range.lower() or 'current' in date_range.lower():
                        end_year = datetime.now().year
                    else:
                        end_year = int(re.search(r'\d{4}$', date_range).group())
                    
                    start_year = int(re.search(r'^\d{4}', date_range).group())
                    if 1980 <= start_year <= datetime.now().year and 1980 <= end_year <= datetime.now().year:
                        experience = end_year - start_year
                    if 0 <= experience <= 50:
                        years.add(experience)
                except (ValueError, AttributeError):
                    continue
        
        # Calculate different types of experience
        if years:
            experience_info['total_years'] = max(years)
            
            # Look for specific experience types
            teaching_pattern = r'(?:teaching|lecturing|instructor|professor).*?(\d+)\s*(?:years?|yrs?)'
            research_pattern = r'(?:research|r&d|development).*?(\d+)\s*(?:years?|yrs?)'
            
            teaching_matches = re.findall(teaching_pattern, text, re.IGNORECASE)
            research_matches = re.findall(research_pattern, text, re.IGNORECASE)
            
            if teaching_matches:
                experience_info['teaching_years'] = max(int(y) for y in teaching_matches if 0 <= int(y) <= 50)
            if research_matches:
                experience_info['research_years'] = max(int(y) for y in research_matches if 0 <= int(y) <= 50)
            
            # Calculate relevant years (max of teaching and research)
            experience_info['relevant_years'] = max(experience_info['teaching_years'], 
                                                  experience_info['research_years'])
        
        return experience_info

    def extract_education(self, text: str) -> Dict[str, List[Dict]]:
        """Extract education information from text"""
        text_lower = self._clean_text(text.lower())
        education_info = {
            'phd': [],
            'masters': [],
            'bachelors': [],
            'diploma': []
        }
        
        # Extract education details
        for level, patterns in self.education_patterns.items():
            for pattern in patterns:
                matches = re.finditer(r'\b' + pattern + r'\b', text_lower)
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text_lower), match.end() + 100)
                    context = text_lower[start:end]
                    
                    # Extract degree details
                    degree_info = {
                        'degree': match.group(),
                        'field': self._extract_field(context),
                        'institution': self._extract_institution(context),
                        'year': self._extract_year(context)
                    }
                    
                    education_info[level].append(degree_info)
        
        return education_info

    def _extract_field(self, text: str) -> str:
        """Extract field of study from text"""
        field_patterns = [
            r'(?:in|of)\s+([a-z\s]+)(?:engineering|science|arts|technology|computer)',
            r'(?:major|specialization|specialized|focused)\s+(?:in|on)\s+([a-z\s]+)',
            r'(?:degree|program)\s+(?:in|of)\s+([a-z\s]+)'
        ]
        
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return ""

    def _extract_institution(self, text: str) -> str:
        """Extract institution name from text"""
        institution_patterns = [
            r'(?:from|at)\s+([a-z\s]+(?:university|college|institute|school))',
            r'(?:graduated|completed|studied)\s+(?:from|at)\s+([a-z\s]+)',
            r'(?:institution|university|college)\s+(?:name)?\s*:?\s*([a-z\s]+)'
        ]
        
        for pattern in institution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return ""

    def _extract_year(self, text: str) -> str:
        """Extract year from text"""
        year_patterns = [
            r'(?:in|year|graduated|completed)\s+(?:19|20)\d{2}',
            r'(?:19|20)\d{2}'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        
        return ""
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from CV text with improved error handling"""
        contact_info = {}
        
        try:
            # Extract email with improved pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                # Validate email format
                email = emails[0].lower()
                if self._is_valid_email(email):
                    contact_info['email'] = email
                else:
                    logger.warning(f"Invalid email format found: {email}")
                    contact_info['email'] = "unknown@example.com"  # Provide default email
            else:
                contact_info['email'] = "unknown@example.com"  # Provide default email
            
            # Extract phone with improved patterns
            phone_patterns = [
                r'\b(?:\+92|0092|92)?\s*\d{3}\s*\d{7}\b',  # Pakistani format
                r'\b\d{3}-\d{3}-\d{4}\b',  # US format
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # US format with parentheses
                r'\b\d{10,11}\b'  # Simple 10-11 digit number
            ]
            
            for pattern in phone_patterns:
                phones = re.findall(pattern, text)
                if phones:
                    contact_info['phone'] = phones[0]
                    break
            
            # Extract location with improved pattern
            location_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            locations = re.findall(location_pattern, text)
            if locations:
                contact_info['location'] = f"{locations[0][0]}, {locations[0][1]}"
            
            # Extract name if not provided
            name_pattern = r'^[\s\n]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
            name_match = re.search(name_pattern, text)
            if name_match:
                contact_info['name'] = name_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Error extracting contact info: {str(e)}")
            contact_info['email'] = "unknown@example.com"  # Ensure email is always present
            return contact_info
        
        return contact_info
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        try:
            # Basic email format validation
            if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', email):
                return False
            
            # Check for common disposable email domains
            disposable_domains = {
                'tempmail.com', 'throwawaymail.com', 'mailinator.com',
                'guerrillamail.com', 'sharklasers.com', 'yopmail.com'
            }
            
            domain = email.split('@')[1].lower()
            if domain in disposable_domains:
                return False
            
            # Check for minimum length requirements
            if len(email) < 5 or len(email) > 254:  # RFC 5321
                return False
            
            return True
            
        except Exception:
            return False

    def extract_certifications(self, text: str) -> List[Dict]:
        """Extract certifications and licenses from text"""
        certifications = []
        text_lower = self._clean_text(text.lower())
        
        for pattern in self.certification_patterns:
            matches = re.finditer(r'\b' + pattern + r'\b', text_lower)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text_lower), match.end() + 50)
                context = text_lower[start:end]
                
                cert_info = {
                    'name': match.group().title(),
                    'issuer': self._extract_certification_issuer(context),
                    'year': self._extract_year(context)
                }
                
                if cert_info not in certifications:
                    certifications.append(cert_info)
        
        return certifications

    def _extract_certification_issuer(self, text: str) -> str:
        """Extract certification issuer from text"""
        issuer_patterns = [
            r'(?:by|from|issued by)\s+([a-z\s]+)',
            r'(?:certified|certification)\s+(?:by|from)\s+([a-z\s]+)',
            r'(?:issuer|provider|organization)\s*:?\s*([a-z\s]+)'
        ]
        
        for pattern in issuer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return ""
    
    def calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """Calculate various text complexity metrics"""
        if not text.strip():
            return {
                'sentences': 0,
                'words': 0,
                'avg_sentence_length': 0,
                'complexity_score': 0,
                'readability_score': 0,
                'vocabulary_diversity': 0
            }
        
        # Basic metrics
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
        else:
            sentences = text.split('.')
            words = text.lower().split()
        
        num_sentences = len(sentences)
        num_words = len(words)
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        
        # Calculate complexity metrics
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / num_words if num_words > 0 else 0
        
        # Calculate readability score (Flesch-Kincaid)
        if num_sentences > 0 and num_words > 0:
            avg_syllables = sum(self._count_syllables(word) for word in words) / num_words
            readability_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
        else:
            readability_score = 0
        
        return {
            'sentences': num_sentences,
            'words': num_words,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complexity_score': round(vocabulary_diversity, 3),
            'readability_score': round(readability_score, 2),
            'vocabulary_diversity': round(vocabulary_diversity, 3)
        }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count
    
    def _is_valid_skill(self, skill: str) -> bool:
        """Check if a term is a valid skill (not in blacklist)"""
        skill_lower = skill.lower()
        
        # Check against blacklist
        if skill_lower in self.skill_blacklist:
            return False
        
        # Check if it contains blacklisted words
        for blacklisted in self.skill_blacklist:
            if blacklisted in skill_lower:
                return False
        
        # Filter out single characters and very short terms
        if len(skill) <= 2 and skill.lower() != 'r':  # Allow 'R' programming language
            return False
        
        # Filter out numeric-only terms
        if skill.isdigit():
            return False
        
        # Filter out terms that are mostly punctuation
        if len(re.sub(r'[^\w\s]', '', skill)) <= 2:
            return False
        
        return True
    
    def _is_technology_term(self, skill: str) -> bool:
        """Check if a term is likely a technology/tool term"""
        skill_lower = skill.lower()
        
        # Technology indicators
        tech_indicators = [
            'framework', 'library', 'language', 'tool', 'platform', 'api',
            'database', 'server', 'engine', 'runtime', 'sdk', 'ide',
            'compiler', 'interpreter', 'version', 'software', 'app',
            'js', 'py', 'java', 'net', 'sql', 'html', 'css', 'xml',
            'json', 'rest', 'http', 'tcp', 'ip', 'web', 'mobile'
        ]
        
        # Check if it contains any technology indicators
        for indicator in tech_indicators:
            if indicator in skill_lower:
                return True
        
        # Check if it's a known file extension or protocol
        if skill_lower.startswith('.') or skill_lower.endswith('.js') or skill_lower.endswith('.py'):
            return True
        
        # If it's very short and contains tech-like characters
        if len(skill) <= 5 and any(char in skill_lower for char in ['++', '#', '.', '-']):
            return True
        
        return False 

    def process_cv(self, cv_text: str, name: str = None) -> Dict:
        """Main CV processing function with improved error handling"""
        try:
            if not cv_text or not cv_text.strip():
                logger.warning("Empty CV text provided")
                return self._empty_profile()
            
            logger.info(f"Processing CV for: {name or 'Unknown'}")
            
            # Clean the text
            cleaned_text = self._clean_text(cv_text)
            
            # Extract all information
            skills = self.extract_skills(cleaned_text)
            
            # Handle experience extraction with error handling
            try:
                experience = self.extract_experience_years(cleaned_text)
                # Ensure experience_years is an integer
                experience['total_years'] = int(experience.get('total_years', 0))
                experience['relevant_years'] = int(experience.get('relevant_years', 0))
                experience['teaching_years'] = int(experience.get('teaching_years', 0))
                experience['research_years'] = int(experience.get('research_years', 0))
            except Exception as e:
                logger.warning(f"Error extracting experience: {str(e)}")
                experience = {
                    'total_years': 0,
                    'relevant_years': 0,
                    'teaching_years': 0,
                    'research_years': 0
                }
            
            education = self.extract_education(cleaned_text)
            contact_info = self.extract_contact_info(cleaned_text)
            certifications = self.extract_certifications(cleaned_text)
            text_metrics = self.calculate_text_complexity(cleaned_text)
            
            processed_data = {
                'name': name or contact_info.get('name', 'Unknown'),
                'contact_info': contact_info,
                'skills': skills,
                'experience': experience,
                'education': education,
                'certifications': certifications,
                'text_metrics': text_metrics,
                'raw_text': cv_text
            }
            
            # Log extraction results
            total_skills = sum(len(s) for s in skills.values())
            total_education = sum(len(e) for e in education.values())
            logger.info(f"Extracted {total_skills} skills, {experience['total_years']} years experience, {total_education} education levels")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing CV: {str(e)}")
            # Return empty profile instead of raising error
            return self._empty_profile()

    def _empty_profile(self) -> Dict:
        """Return an empty profile structure"""
        return {
            'name': 'Unknown',
            'contact_info': {},
            'skills': {category: [] for category in self.skill_patterns.keys()},
            'experience': {
                'total_years': 0,
                'relevant_years': 0,
                'teaching_years': 0,
                'research_years': 0
            },
            'education': {
                'phd': [],
                'masters': [],
                'bachelors': [],
                'diploma': []
            },
            'certifications': [],
            'text_metrics': {
                'sentences': 0,
                'words': 0,
                'avg_sentence_length': 0,
                'complexity_score': 0,
                'readability_score': 0,
                'vocabulary_diversity': 0
            },
            'raw_text': ''
        }

    def _clean_skill_name(self, skill: str) -> str:
        """Clean and normalize skill names"""
        # Remove extra whitespace and convert to title case
        skill = re.sub(r'\s+', ' ', skill.strip())
        
        # Handle special cases
        special_cases = {
            'c++': 'C++',
            'c#': 'C#',
            'javascript': 'JavaScript',
            'js': 'JavaScript',
            'typescript': 'TypeScript',
            'ts': 'TypeScript',
            'nodejs': 'Node.js',
            'node.js': 'Node.js',
            'reactjs': 'React',
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'aws': 'AWS',
            'azure': 'Azure',
            'gcp': 'Google Cloud',
            'google cloud': 'Google Cloud',
            'html5': 'HTML5',
            'css3': 'CSS3',
            'jsx': 'JSX',
            'tsx': 'TSX',
            'json': 'JSON',
            'xml': 'XML',
            'yaml': 'YAML',
            'toml': 'TOML',
            'markdown': 'Markdown',
            'rest': 'REST',
            'graphql': 'GraphQL',
            'soap': 'SOAP',
            'http': 'HTTP',
            'https': 'HTTPS',
            'tcp': 'TCP',
            'ip': 'IP',
            'udp': 'UDP',
            'dns': 'DNS',
            'ssh': 'SSH',
            'ssl': 'SSL',
            'tls': 'TLS',
            'api': 'API',
            'sdk': 'SDK',
            'ide': 'IDE',
            'cli': 'CLI',
            'gui': 'GUI',
            'ui': 'UI',
            'ux': 'UX',
            'ci': 'CI',
            'cd': 'CD',
            'devops': 'DevOps',
            'agile': 'Agile',
            'scrum': 'Scrum',
            'kanban': 'Kanban',
            'tdd': 'TDD',
            'bdd': 'BDD',
            'oop': 'OOP',
            'fp': 'FP',
        }
        
        skill = special_cases.get(skill.lower(), skill)
        
        return skill.title()

    def match_cv_with_jd(self, cv_data: Dict, jd_text: str) -> Dict:
        """Match CV with job description and calculate match score"""
        # Extract skills from JD
        jd_skills = self.extract_skills(jd_text)
        
        # Calculate skill match
        skill_matches = {}
        for category in self.skill_patterns.keys():
            cv_skills = set(cv_data['skills'].get(category, []))
            jd_category_skills = set(jd_skills.get(category, []))
            
            if jd_category_skills:
                match_score = len(cv_skills.intersection(jd_category_skills)) / len(jd_category_skills)
                skill_matches[category] = {
                    'match_score': round(match_score * 100, 2),
                    'matched_skills': list(cv_skills.intersection(jd_category_skills)),
                    'missing_skills': list(jd_category_skills - cv_skills)
                }
        
        # Calculate overall match score
        overall_score = sum(match['match_score'] for match in skill_matches.values()) / len(skill_matches) if skill_matches else 0
        
        return {
            'overall_match_score': round(overall_score, 2),
            'skill_matches': skill_matches,
            'cv_data': cv_data
        }

    def analyze_cv_quality(self, cv_data: Dict) -> Dict:
        """Analyze the quality of a CV"""
        quality_metrics = {
            'completeness': 0,
            'readability': 0,
            'professionalism': 0,
            'overall_score': 0
        }
        
        # Calculate completeness score
        required_sections = ['skills', 'experience', 'education', 'contact_info']
        completeness_scores = []
        
        for section in required_sections:
            if section in cv_data and cv_data[section]:
                if section == 'skills':
                    # Check if there are skills in any category
                    has_skills = any(len(skills) > 0 for skills in cv_data['skills'].values())
                    completeness_scores.append(100 if has_skills else 0)
                elif section == 'experience':
                    # Check if there's any experience
                    has_experience = cv_data['experience']['total_years'] > 0
                    completeness_scores.append(100 if has_experience else 0)
                elif section == 'education':
                    # Check if there's any education
                    has_education = any(len(edu) > 0 for edu in cv_data['education'].values())
                    completeness_scores.append(100 if has_education else 0)
                elif section == 'contact_info':
                    # Check if there's at least an email
                    has_contact = 'email' in cv_data['contact_info']
                    completeness_scores.append(100 if has_contact else 0)
            else:
                completeness_scores.append(0)
        
        quality_metrics['completeness'] = round(sum(completeness_scores) / len(required_sections), 2)
        
        # Calculate readability score
        text_metrics = cv_data.get('text_metrics', {})
        if text_metrics:
            readability_score = text_metrics.get('readability_score', 0)
            # Normalize readability score to 0-100
            quality_metrics['readability'] = round(max(0, min(100, readability_score)), 2)
        
        # Calculate professionalism score
        professionalism_scores = []
        
        # Check for proper formatting
        if cv_data.get('raw_text'):
            # Check for consistent formatting
            has_consistent_formatting = not bool(re.search(r'\n{3,}', cv_data['raw_text']))
            professionalism_scores.append(100 if has_consistent_formatting else 0)
        
        # Check for professional email
        if 'email' in cv_data.get('contact_info', {}):
            email = cv_data['contact_info']['email']
            is_professional_email = not any(term in email.lower() for term in ['hotmail', 'yahoo', 'gmail'])
            professionalism_scores.append(100 if is_professional_email else 0)
        
        # Check for certifications
        if cv_data.get('certifications'):
            professionalism_scores.append(min(100, len(cv_data['certifications']) * 20))
        
        quality_metrics['professionalism'] = round(sum(professionalism_scores) / len(professionalism_scores), 2) if professionalism_scores else 0
        
        # Calculate overall score
        quality_metrics['overall_score'] = round(
            (quality_metrics['completeness'] * 0.4 +
             quality_metrics['readability'] * 0.3 +
             quality_metrics['professionalism'] * 0.3),
            2
        )
        
        return quality_metrics
        