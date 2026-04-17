# 云服务器部署说明

这份说明只对应现在的最简部署结构：

- 一份编排文件：`docker-compose.yml`
- 一份环境变量模板：`.env.example`

你以后不需要再区分“本地 compose / 生产 compose / 服务器 env 模板”了。

## 1. 先理解现在的结构

这个项目在 Docker 里会启动 4 个服务：

- `frontend`：前端页面，对外提供访问入口
- `backend`：后端 API，只在 Docker 内部网络提供服务
- `postgres`：PostgreSQL + PGVector
- `mongo`：MongoDB

现在默认只有前端端口对外暴露：

- 浏览器访问：`http://服务器IP`
- 接口文档访问：`http://服务器IP/api/docs`

下面这些端口默认不会直接暴露到公网：

- `5432`
- `27017`
- `1016`

这样比之前更简单，也更安全。

## 2. 推荐服务器配置

建议至少：

- `4 vCPU`
- `8 GB RAM`
- `50 GB` 可用磁盘

更稳妥一些：

- `4 vCPU`
- `8~16 GB RAM`
- `80 GB` 可用磁盘

原因很简单：这个项目的 Python 依赖比较重，Docker 构建会占不少空间。

## 3. 安装基础环境

推荐系统：

- Ubuntu 22.04
- Ubuntu 24.04

安装 Docker：

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

执行完最后一条后，重新登录一次服务器。

## 4. 拉取代码

```bash
git clone https://github.com/edenxie-xgk/enterprise-repository.git
cd enterprise-repository
git checkout codex/backend-smoke-gate
```

如果你后面已经把代码合并到主分支，就不需要 `git checkout` 这一步。

## 5. 准备环境变量

复制模板：

```bash
cp .env.example .env
```

然后编辑：

```bash
nano .env
```

第一次部署时，先只改这些值：

- `JWT_SECRET_KEY`
- `CORS_ALLOW_ORIGINS`
- `DOCKER_MILVUS_URI`
- `FRONTEND_PORT`
- `DOCKER_APP_ENV`
- `APP_ENV`
- `POSTGRES_PASSWORD`
- `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`

如果你是第一次上手，可以把 `.env.example` 里的配置理解成两层：

- 第一层：文件最上面“First look here”那几项，这是第一次部署最需要理解的
- 第二层：后面的默认值，大多数情况下先保持不动就行

## 6. 服务器最常用的 `.env` 改法

如果你是在云服务器上部署，通常推荐这样改：

```env
FRONTEND_PORT=80
DOCKER_APP_ENV=production
APP_ENV=production
DEBUG=false
```

如果你先用服务器公网 IP 访问前端：

```env
CORS_ALLOW_ORIGINS=http://你的服务器公网IP
```

如果你后面绑定域名：

```env
CORS_ALLOW_ORIGINS=https://your-domain.com
```

Milvus 地址要改成“容器能够访问到的真实地址”，例如：

```env
DOCKER_MILVUS_URI=http://10.0.0.12:19530
```

如果 Milvus 不是跑在这台服务器宿主机上，就不要再用：

```env
http://host.docker.internal:19530
```

## 7. 启动项目

直接执行：

```bash
docker compose up -d --build
```

这条命令会完成：

1. 启动 PostgreSQL
2. 启动 MongoDB
3. 构建并启动后端
4. 构建并启动前端

后端容器启动时还会自动：

- 等待数据库就绪
- 执行 `alembic upgrade head`
- 启动 FastAPI

## 8. 如何检查是否启动成功

看容器状态：

```bash
docker compose ps
```

看后端日志：

```bash
docker compose logs -f backend
```

看前端日志：

```bash
docker compose logs -f frontend
```

## 9. 启动成功后怎么访问

前端：

```text
http://你的服务器公网IP
```

接口文档：

```text
http://你的服务器公网IP/api/docs
```

因为后端没有直接暴露到公网，所以文档入口也走前端反向代理。

## 10. 云服务器安全组怎么开

至少放行：

- `22`：SSH
- `80`：HTTP

如果以后做 HTTPS，再放行：

- `443`

不建议放行：

- `5432`
- `27017`
- `1016`

## 11. 常见问题

### 1) 页面能打开，但接口不通

先看后端日志：

```bash
docker compose logs -f backend
```

重点检查：

- `.env` 里的模型 API Key
- PostgreSQL 账号密码
- `DOCKER_MILVUS_URI`

### 2) 后端起不来，提示数据库连不上

常见原因：

- `.env` 里的 `POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD` 填错
- 服务器磁盘空间不足
- 首次构建尚未完成

### 3) 长期记忆没生效

先确认这些配置都打开了：

```env
MEMORY_ENABLED=true
MEMORY_BACKEND=milvus
MEMORY_WRITE_ENABLED=true
```

然后再检查：

```env
DOCKER_MILVUS_URI=http://your-milvus-host:19530
```

最后看后端日志里有没有 Milvus 连接错误。

## 12. 最适合新手的部署顺序

我建议你以后每次都按这个顺序来：

1. 登录服务器
2. 拉代码
3. 复制 `.env.example` 为 `.env`
4. 只修改最关键的环境变量
5. 执行 `docker compose up -d --build`
6. 看 `docker compose ps`
7. 看 `docker compose logs -f backend`
8. 浏览器访问服务器 IP

这样最不容易乱。
