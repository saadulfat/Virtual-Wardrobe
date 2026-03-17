# Virtual Wardrobe Application

A comprehensive virtual wardrobe application that allows users to manage their clothing items, create outfit combinations, and use AI-powered style recommendations.

## Features

- **User Authentication**: Secure sign up and login system
- **Wardrobe Management**: Upload and organize clothing items with categories (shirts, pants, shoes, etc.)
- **Virtual Try-On**: Preview how outfits look on a digital model
- **Style Assistant**: AI-powered recommendations using Google Generative AI
- **Weather-Based Suggestions**: Outfit recommendations based on current weather
- **Outfit Combinations**: Smart pairing of clothing items
- **Chatbot Integration**: Interactive style advice

## Prerequisites

- Docker and Docker Compose
- Git

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Environment Configuration

Copy the example environment file and customize your settings:

```bash
copy .env.example .env   # On Windows
# OR
cp .env.example .env     # On macOS/Linux
```

Edit the `.env` file and add your actual API keys and database credentials.
**Important**: Never commit your `.env` file to version control as it contains sensitive information.

### 3. Docker Setup (Recommended)

The easiest way to run this application is using Docker. Make sure you have Docker and Docker Compose installed.

1. Ensure your `.env` file is properly configured with your API keys
2. From the project root directory, run:
   ```bash
   docker-compose up -d
   ```
3. The application will be available at `http://localhost:8000`
4. To stop the application:
   ```bash
   docker-compose down
   ```

### 4. Alternative: Local Setup

If you prefer to run locally instead of using Docker:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure MySQL is running and accessible

5. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```

### 5. Required Environment Variables

The application requires the following environment variables to be set in your `.env` file:

- `DATABASE_URL` - Database connection string (e.g., `mysql+pymysql://root:yourpassword@localhost:3306/wardrobe`)
- `SECRET_KEY` - Secret key for session management (use a strong random string)
- `KIE_API_KEY` - API key for KIE service
- `WEATHER_API_KEY` - OpenWeatherMap API key
- `STABILITY_API_KEY` - Stability AI API key for image generation
- `GEMINI_API_KEY` - Google Gemini API key for AI features

### 6. Run with Docker

Build and start the application:

```bash
docker-compose up -d
```

The application will be available at `http://localhost:8000`.

### 7. Alternative: Local Setup

If you prefer to run locally instead of using Docker:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure MySQL is running and accessible

5. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

1. Visit `http://localhost:8000` in your browser
2. Sign up for a new account or log in if you already have one
3. Upload your clothing items to your wardrobe
4. Create outfit combinations
5. Use the style assistant for AI-powered recommendations
6. Try on outfits virtually using the try-on feature

## Project Structure

```
virtual-wardrobe/
├── app/
│   ├── static/          # Static assets (CSS, images, uploads)
│   ├── templates/       # HTML templates
│   ├── main.py          # Main FastAPI application
│   ├── database.py      # Database configuration
│   ├── models.py        # Database models
│   ├── color_detection.py # Color detection utilities
│   └── chatbot_service.py # AI chatbot integration
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
├── .env.example        # Example environment variables
└── README.md           # This file
```

## API Endpoints

- `GET /` - Home page
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /signup` - Registration page
- `POST /signup` - Register new user
- `GET /wardrobe` - View wardrobe items
- `POST /upload-outfit` - Upload new clothing items
- `GET /try-on` - Virtual try-on interface
- `POST /generate-outfit` - AI outfit generation
- `GET /chat` - Chatbot interface
- `GET /weather-outfits` - Weather-based outfit suggestions

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Ensure MySQL is running and accessible
   - Check that the `DATABASE_URL` in your `.env` file is correct
   - Verify database credentials

2. **API Keys Not Working**:
   - Confirm all required API keys are correctly set in the `.env` file
   - Check that API keys have not expired or been revoked

3. **Docker Issues**:
   - Make sure Docker and Docker Compose are properly installed
   - Check Docker logs with `docker-compose logs app` for detailed error information

### Resetting the Application

To reset the application and clear all data:

```bash
docker-compose down -v  # Removes containers and volumes
docker-compose up -d    # Restarts with fresh data
```

## Security Best Practices

- Store all API keys and sensitive information in the `.env` file
- Never commit the `.env` file to version control
- Use strong passwords and secure API keys
- Regularly rotate API keys
- Keep dependencies up to date

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please open an issue in the repository or contact the project maintainers.