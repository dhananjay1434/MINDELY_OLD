# AI Friend Application Deployment Guide

This guide explains how to deploy the AI Friend application with the frontend on Netlify and the backend on a free Python hosting service.

## Project Structure

```
.
├── backend/
│   ├── app.py              # Flask API backend
│   ├── Procfile            # For Heroku/Render deployment
│   ├── requirements.txt    # Python dependencies
│   ├── runtime.txt         # Python version specification
│   └── .env                # Environment variables (not committed to git)
│
└── frontend/
    ├── index.html          # Main HTML file
    ├── styles.css          # CSS styles
    ├── app.js              # Frontend JavaScript
    └── netlify.toml        # Netlify configuration
```

## Backend Deployment (Render.com)

1. **Create a Render.com account**
   - Go to [render.com](https://render.com/) and sign up for a free account

2. **Create a new Web Service**
   - Click "New" and select "Web Service"
   - Connect your GitHub repository or use the "Upload" option
   - Select the repository containing your backend code

3. **Configure the Web Service**
   - Name: `ai-friend-backend` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Select the Free plan

4. **Set Environment Variables**
   - Add the following environment variables:
     - `GEMINI_API_KEY`: Your Google Gemini API key

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the deployment to complete
   - Note the URL provided by Render (e.g., `https://ai-friend-backend.onrender.com`)

## Frontend Deployment (Netlify)

1. **Update API URL**
   - Open `frontend/app.js`
   - Update the `API_URL` constant with your backend URL:
     ```js
     const API_URL = 'https://your-backend-url.onrender.com/api';
     ```

2. **Create a Netlify account**
   - Go to [netlify.com](https://netlify.com/) and sign up for a free account

3. **Deploy via Netlify UI**
   - Go to the Netlify dashboard
   - Click "Add new site" > "Deploy manually"
   - Drag and drop your `frontend` folder to the upload area
   - Wait for the deployment to complete
   - Your site will be available at a Netlify subdomain (e.g., `https://your-site-name.netlify.app`)

4. **Custom Domain (Optional)**
   - In the Netlify dashboard, go to "Domain settings"
   - Click "Add custom domain" and follow the instructions

## Alternative Backend Deployment Options

### PythonAnywhere

1. **Create a PythonAnywhere account**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com/) and sign up for a free account

2. **Upload your backend code**
   - Go to the "Files" tab
   - Create a new directory for your project
   - Upload your backend files

3. **Set up a virtual environment**
   - Go to the "Consoles" tab and open a Bash console
   - Navigate to your project directory
   - Create and activate a virtual environment:
     ```bash
     mkvirtualenv --python=/usr/bin/python3.10 myenv
     pip install -r requirements.txt
     ```

4. **Configure a web app**
   - Go to the "Web" tab
   - Click "Add a new web app"
   - Select "Flask" and your Python version
   - Set the path to your Flask app (e.g., `/home/yourusername/myproject/app.py`)
   - Set the working directory to your project folder

5. **Set environment variables**
   - In the "Web" tab, under "WSGI configuration file"
   - Add your environment variables at the top of the file:
     ```python
     import os
     os.environ['GEMINI_API_KEY'] = 'your_api_key_here'
     ```

6. **Reload the web app**
   - Click the "Reload" button to apply changes

### Railway.app

1. **Create a Railway account**
   - Go to [railway.app](https://railway.app/) and sign up

2. **Create a new project**
   - Click "New Project" > "Deploy from GitHub repo"
   - Connect your GitHub repository
   - Select the repository containing your backend code

3. **Configure environment variables**
   - Go to the "Variables" tab
   - Add your environment variables:
     - `GEMINI_API_KEY`: Your Google Gemini API key

4. **Deploy**
   - Railway will automatically deploy your application
   - Note the URL provided by Railway

## Connecting Frontend to Backend

After deploying both the frontend and backend, make sure to update the `API_URL` in `frontend/app.js` with your actual backend URL. If you've already deployed the frontend, you'll need to redeploy it with the updated URL.

## Troubleshooting

- **CORS Issues**: If you encounter CORS errors, make sure your backend has CORS properly configured.
- **API Key Issues**: Verify that your Gemini API key is correctly set in the backend environment variables.
- **Deployment Failures**: Check the deployment logs for specific error messages.

## Maintenance

- Regularly check for security updates
- Monitor your application's performance
- Keep your API keys secure and rotate them periodically
