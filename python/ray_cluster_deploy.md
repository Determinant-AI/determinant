## Deploying the KubeRay operator

### Create the KubeRay operator and all of the resources it needs.

```
kubectl create -k deployment/config/default
```

Confirm that the operator is running in the namespace `ray-system`
```
kubectl -n ray-system get pod --selector=app.kubernetes.io/component=kuberay-operator
```

## Deploying a Ray Cluster
Once the KubeRay operator is running, we are ready to deploy a Ray cluster. To do so, we create a RayCluster Custom Resource (CR).

Deploy a sample Ray Cluster CR from the KubeRay repo:
```
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/samples/ray-cluster.autoscaler.yaml
```

View the created RayCluster CR:
```
kubectl get raycluster
```

View the pods in the Ray cluster
```
kubectl get pods --selector=ray.io/cluster=raycluster-autoscaler
```

observe the pods’ status in real-time, run in a separate shell:
```
watch -n 1 kubectl get pod
```

## Running Applications on a Ray Cluster
identify the Ray cluster's head pod:
```
kubectl get pods --selector=ray.io/cluster=raycluster-autoscaler --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers
```

run a Ray program on the head pod:
```
kubectl exec raycluster-autoscaler-head-xxxxx -it -c ray-head -- python -c "import ray; ray.init()"
```

OR 
```
head_pod=$(kubectl get pods --selector=ray.io/cluster=raycluster-autoscaler --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)

kubectl exec $head_pod -it -c ray-head -- python -c "import ray; ray.init()"
```

## Ray job submission
List all services with IP and ports
```
kubectl get services
```

identify the Ray head service:
```
kubectl get service raycluster-autoscaler-head-svc
```

port-forwarding in a separate shell
```
kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265
```

For production use-cases, you would typically either:
- Access the service from within the Kubernetes cluster or
- Use an ingress controller to expose the service outside the cluster.

```
ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

## Delete Ray cluster
```
kubectl delete raycluster raycluster-autoscaler
```

## Deleting the KubeRay operator
In typical operation, the KubeRay operator should be left as a long-running process that manages many Ray clusters. If you would like to delete the operator and associated resources, run
```
kubectl delete -k ray/kuberay/ray-operator/config/default
```

## Put secrets on K8s Secrets
```
kubectl create secret generic slack-secrets \
  --from-literal=SLACK_API_TOKEN="your_slack_api_token" \
  --from-literal=SLACK_SIGNING_SECRET="your_slack_signing_secret"
```