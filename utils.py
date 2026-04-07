import requests
import json
import re
from datetime import datetime, timedelta

# Todoist API Tools
class TodoistTools:
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://api.todoist.com/rest/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    def get_projects(self):
        """Get all projects from Todoist"""
        response = requests.get(f"{self.base_url}/projects", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get projects: {response.status_code}"}
    
    def get_project(self, project_name):
        """Get a specific project by name"""
        projects = self.get_projects()
        if "error" in projects:
            return projects
        
        for project in projects:
            if project["name"].lower() == project_name.lower():
                return project
        
        return None
    
    def create_project(self, project_name, color="berry_red"):
        """Create a new project in Todoist"""
        # Check if project already exists
        existing_project = self.get_project(project_name)
        if existing_project:
            return existing_project
        
        data = {
            "name": project_name,
            "color": color
        }
        
        response = requests.post(
            f"{self.base_url}/projects", 
            headers=self.headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to create project: {response.status_code}"}
    
    def get_collaborators(self, project_id):
        """Get all collaborators for a project"""
        response = requests.get(
            f"{self.base_url}/projects/{project_id}/collaborators", 
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get collaborators: {response.status_code}"}
    
    def create_task(self, content, project_id, due_string=None, priority=3, assignee_id=None):
        """Create a new task in Todoist"""
        data = {
            "content": content,
            "project_id": project_id,
            "priority": priority
        }
        
        if due_string:
            data["due_string"] = due_string
            
        if assignee_id:
            data["assignee_id"] = assignee_id
        
        response = requests.post(
            f"{self.base_url}/tasks", 
            headers=self.headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to create task: {response.status_code}"}
    
    def create_and_assign_task(self, content, project_name, assignee_name=None, due_string=None, priority=3):
        """Create a task and assign it to a team member"""
        # Get or create project
        project = self.get_project(project_name)
        if not project:
            project = self.create_project(project_name)
            if "error" in project:
                return project
        
        project_id = project["id"]
        
        # If assignee is specified, get their ID
        assignee_id = None
        if assignee_name:
            collaborators = self.get_collaborators(project_id)
            if "error" not in collaborators:
                for collaborator in collaborators:
                    if collaborator["name"].lower() == assignee_name.lower():
                        assignee_id = collaborator["id"]
                        break
        
        # Create the task
        return self.create_task(content, project_id, due_string, priority, assignee_id)

# Transcript Extraction Tool
class TranscriptExtractor:
    def __init__(self, source_type="google_meet"):
        self.source_type = source_type
    
    def get_transcript(self, meeting_id):
        """Extract transcript from the specified source"""
        if self.source_type == "google_meet":
            return self._get_google_meet_transcript(meeting_id)
        elif self.source_type == "whatsapp":
            return self._get_whatsapp_transcript(meeting_id)
        elif self.source_type == "telegram":
            return self._get_telegram_transcript(meeting_id)
        else:
            return {"error": f"Unsupported source type: {self.source_type}"}
    
    def _get_google_meet_transcript(self, meeting_id):
        """Get transcript from Google Meet"""
        # This is a placeholder. In a real implementation, you would use the Google Meet API
        # or a service that provides access to Google Meet transcripts
        
        # For demo purposes, we'll return a sample transcript
        return {
            "meeting_id": meeting_id,
            "transcript": "This is a sample transcript from Google Meet. We need to create a new project called Marketing Campaign and assign tasks to the team.",
            "participants": ["John Smith", "Jane Doe", "Bob Johnson"]
        }
    
    def _get_whatsapp_transcript(self, chat_id):
        """Get transcript from WhatsApp"""
        # This is a placeholder. In a real implementation, you would use the WhatsApp Business API
        # or a service that provides access to WhatsApp chat history
        
        # For demo purposes, we'll return a sample transcript
        return {
            "chat_id": chat_id,
            "transcript": "This is a sample transcript from WhatsApp. We need to create a new project called Website Redesign and assign tasks to the team.",
            "participants": ["John Smith", "Jane Doe", "Bob Johnson"]
        }
    
    def _get_telegram_transcript(self, chat_id):
        """Get transcript from Telegram"""
        # This is a placeholder. In a real implementation, you would use the Telegram Bot API
        # to access chat history
        
        # For demo purposes, we'll return a sample transcript
        return {
            "chat_id": chat_id,
            "transcript": "This is a sample transcript from Telegram. We need to create a new project called Product Launch and assign tasks to the team.",
            "participants": ["John Smith", "Jane Doe", "Bob Johnson"]
        }

# Telegram Communication Tool
class TelegramCommunicator:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message):
        """Send a message to the Telegram group"""
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(
            f"{self.base_url}/sendMessage", 
            data=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to send message: {response.status_code}"}
    
    def ask_confirmation(self, question, options=None):
        """Ask for confirmation with inline keyboard buttons"""
        if options is None:
            options = ["Yes", "No"]
            
        keyboard = []
        for option in options:
            keyboard.append([{"text": option, "callback_data": option}])
            
        data = {
            "chat_id": self.chat_id,
            "text": question,
            "reply_markup": json.dumps({
                "inline_keyboard": keyboard
            })
        }
        
        response = requests.post(
            f"{self.base_url}/sendMessage", 
            data=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to send message: {response.status_code}"}

# AI Task Extractor
class TaskExtractor:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_tasks_from_transcript(self, transcript):
        """Extract tasks from transcript using LLM"""
        prompt = f"""
        Please analyze the following meeting transcript and identify:
        1. Project names mentioned
        2. Tasks that need to be completed
        3. Who should be assigned to each task (if mentioned)
        4. Due dates for tasks (if mentioned)
        
        Format your response as JSON with the following structure:
        {{
            "projects": [
                {{
                    "name": "Project Name",
                    "tasks": [
                        {{
                            "content": "Task description",
                            "assignee": "Assignee Name or null",
                            "due_string": "Due date string or null",
                            "priority": 1-4 (4 is highest)
                        }}
                    ]
                }}
            ]
        }}
        
        Transcript:
        {transcript}
        """
        
        response = self.llm.invoke(prompt).content
        
        # Extract JSON from response (in case LLM adds explanatory text)
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        try:
            # Clean up the string and parse JSON
            clean_json = json_str.strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            return {"error": "Failed to parse extracted tasks"}

# Meeting Manager Agent
class TodoistMeetingManager:
    def __init__(self, todoist_api_token, telegram_bot_token=None, telegram_chat_id=None, transcript_source="google_meet", llm=None):
        self.todoist_tools = TodoistTools(todoist_api_token)
        self.transcript_extractor = TranscriptExtractor(transcript_source)
        self.telegram = None
        if telegram_bot_token and telegram_chat_id:
            self.telegram = TelegramCommunicator(telegram_bot_token, telegram_chat_id)
        self.task_extractor = TaskExtractor(llm)
    
    def process_meeting(self, meeting_id):
        """Process a meeting and create tasks in Todoist"""
        # Get transcript
        transcript_data = self.transcript_extractor.get_transcript(meeting_id)
        if "error" in transcript_data:
            return transcript_data
        
        # Extract tasks
        extracted_data = self.task_extractor.extract_tasks_from_transcript(transcript_data["transcript"])
        if "error" in extracted_data:
            return extracted_data
        
        results = {
            "projects_created": [],
            "tasks_created": []
        }
        
        # Create projects and tasks
        for project_data in extracted_data["projects"]:
            project_name = project_data["name"]
            
            # Ask for confirmation if Telegram is enabled
            if self.telegram:
                confirmation = self.telegram.ask_confirmation(
                    f"Should I create a new project '{project_name}'?"
                )
                # In a real implementation, you would wait for the response
                # For demo purposes, we'll assume confirmation is received
            
            # Create project
            project = self.todoist_tools.create_project(project_name)
            if "error" in project:
                results["error"] = project["error"]
                return results
            
            results["projects_created"].append(project_name)
            
            # Create tasks
            for task_data in project_data["tasks"]:
                task = self.todoist_tools.create_and_assign_task(
                    task_data["content"],
                    project_name,
                    task_data.get("assignee"),
                    task_data.get("due_string"),
                    task_data.get("priority", 3)
                )
                
                if "error" in task:
                    results["task_errors"] = results.get("task_errors", []) + [task["error"]]
                else:
                    results["tasks_created"].append({
                        "content": task_data["content"],
                        "project": project_name,
                        "assignee": task_data.get("assignee")
                    })
                    
                    # Notify team member if Telegram is enabled
                    if self.telegram and task_data.get("assignee"):
                        self.telegram.send_message(
                            f"*New Task Assigned*\n\n"
                            f"Project: {project_name}\n"
                            f"Task: {task_data['content']}\n"
                            f"Assigned to: {task_data['assignee']}\n"
                            f"Due: {task_data.get('due_string', 'Not specified')}"
                        )
        
        return results
