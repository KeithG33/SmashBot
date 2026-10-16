# SmashBot as a UMI bot. From the SmashBot repo root, with the umi library and a directory of slim
# checkpoints (python -m umi_bot.slim) as named contexts:
#   docker build -f umi_bot/Dockerfile \
#     --build-context umi=../computer-melee-championship/umi --build-context models=<dir> \
#     --build-arg CHECKPOINT=/models/qk-1057011.pt --build-arg DELAY=21 -t smashbot:dev .
#   umi-conform --image smashbot:dev
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir numpy dm-tree melee pyarrow tqdm py7zr
COPY --from=umi . /opt/umi
RUN pip install --no-cache-dir /opt/umi
# SmashBot's slippi_ai copy: only its data types and embeddings are used, so none of its TF dependencies
COPY vendor/slippi-ai /opt/slippi-ai
RUN pip install --no-cache-dir --no-deps /opt/slippi-ai
COPY smashbot /bot/smashbot
COPY umi_bot /bot/umi_bot
COPY --from=models . /models
WORKDIR /bot
ARG CHECKPOINT=/models/qk.pt
ARG DELAY=21
ARG SCOPE=12
ARG NAME=smashbot
ENV CHECKPOINT=${CHECKPOINT} DELAY=${DELAY} SCOPE=${SCOPE} BOT_NAME=${NAME} HOME=/home/bot TRITON_CACHE_DIR=/tmp/triton
USER 10001
CMD ["sh", "-c", "exec python -m umi_bot.bot --checkpoint \"$CHECKPOINT\" --delay \"$DELAY\" --scope \"$SCOPE\" --name \"$BOT_NAME\""]
