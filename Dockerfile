FROM apify/actor-node:20

COPY package*.json ./
RUN npm install --omit=dev --omit=optional

COPY . ./

COPY README.md ./
CMD ["node", "dist/main.js"]
