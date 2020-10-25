Build image:
```
docker build --tag benchmark-tvm:1.0 .
```

Create container:
```
docker run -v="$(pwd)":/workspace -w=/workspace -it benchmark-tvm:1.0 bash
```

Enter container:
```
docker exec -w=/workspace -it container_name bash
```