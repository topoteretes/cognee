# Use an official Node.js runtime as a parent image
FROM node:18.17.0

# Set the working directory to /app
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install any needed packages specified in package.json
RUN npm install

# Copy the rest of the application code to the working directory
COPY . .

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run npm install and then npm run dev
CMD ["sh", "-c", "npm install && npm run dev"]