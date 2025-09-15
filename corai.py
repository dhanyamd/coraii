import os
import re 
import json
import textwrap 
import base64 
from pathlib import Path 
from typing import List, Dict, Any, Optional, Union 
from together import Together

reasoning_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"