FROM apify/actor-node:20

COPY package*.json ./
RUN npm install --include=dev --omit=optional

COPY tsconfig.json ./
COPY src/ ./src/
COPY .actor/ ./.actor/

RUN npx tsc && npm prune --omit=dev --omit=optional

COPY README.md ./
CMD ["node", "dist/main.js"]
