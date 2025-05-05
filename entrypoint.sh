# Allow Docker to access your X session
xhost +local:root

# Run Compose
docker-compose up --build
