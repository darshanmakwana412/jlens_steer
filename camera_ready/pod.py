"""RunPod helper for the camera-ready experiments: create | status | ssh | delete (RTX PRO 4500)."""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
STATE = HERE / ".pod.json"
for line in open(os.path.expanduser("~/Projects/.env")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"'))
KEY = os.environ["DARSHAN_RUNPOD_API_KEY"]
GPU = "NVIDIA RTX PRO 4500 Blackwell"


def gql(q, variables=None):
    out = subprocess.run(["curl", "-s", "-A", "curl/8.7.1", "-H", "Content-Type: application/json",
                          "-H", f"Authorization: Bearer {KEY}", "https://api.runpod.io/graphql",
                          "-d", json.dumps({"query": q, "variables": variables or {}})],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    if d.get("errors"):
        raise SystemExit(json.dumps(d["errors"], indent=1))
    return d["data"]


def create():
    pub = open(os.path.expanduser("~/.ssh/runpod_key.pub")).read().strip()
    for gpu in [GPU, GPU + " Server Edition"]:
        inp = {"cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": gpu, "name": "jspace-camera-ready",
               "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "containerDiskInGb": 80,
               "volumeInGb": 0, "ports": "22/tcp", "supportPublicIp": True,
               "env": [{"key": "PUBLIC_KEY", "value": pub}, {"key": "HF_TOKEN", "value": os.environ.get("HF_TOKEN", "")}]}
        try:
            pod = gql("mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) "
                      "{ id costPerHr machine { gpuDisplayName location } } }", {"i": inp})["podFindAndDeployOnDemand"]
        except SystemExit as e:
            print(gpu, "failed:", e)
            continue
        STATE.write_text(json.dumps(pod, indent=1))
        print(json.dumps(pod, indent=1))
        return
    sys.exit("no capacity")


def info():
    s = json.loads(STATE.read_text())
    return gql("query($id: String!) { pod(input: {podId: $id}) { id desiredStatus costPerHr runtime { uptimeInSeconds "
               "ports { ip isIpPublic privatePort publicPort type } } } }", {"id": s["id"]})["pod"]


def status():
    print(json.dumps(info(), indent=1))
    print(json.dumps(gql("{ myself { clientBalance currentSpendPerHr } }"), indent=1))


def ssh():
    p = info()
    port = [x for x in p["runtime"]["ports"] if x["privatePort"] == 22 and x["isIpPublic"]][0]
    print(f"ssh -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -p {port['publicPort']} root@{port['ip']}")


def delete():
    s = json.loads(STATE.read_text())
    print(gql("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": s["id"]}))


{"create": create, "status": status, "ssh": ssh, "delete": delete}[sys.argv[1]]()
