# Use an official Node.js runtime as a parent image
FROM node:22-alpine

# Set the working directory to /app
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package.json package-lock.json ./

# Install any needed packages specified in package.json
RUN npm ci
RUN npm rebuild lightningcss

# Copy the rest of the application code to the working directory
COPY src ./src
COPY public ./public
COPY next.config.ts .
COPY postcss.config.mjs .
COPY tsconfig.json .
# Local mode is opt-in (NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT=false); without this
# the app boots in cloud mode and expects Auth0.
COPY .env.local .

# Build the app and run it
CMD ["npm", "run", "dev"]
