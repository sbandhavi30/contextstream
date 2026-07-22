"""
scenario.py — Synthetic SRE incident data for the demo.

Simulates a production OOM incident with realistic tool outputs:
  Tool 1: kubectl describe → OOMKilled pod
  Tool 2: sql query      → top memory consumers in orders table
  Tool 3: bash df        → disk pressure on kubelet node
  Tool 4: file read      → deployment manifest with low memory limit

The agent task: identify root cause and recommend fix.
"""

KUBECTL_OUTPUT = """\
Name:               web-backend-6f9d7c4-xkv2p
Namespace:          production
Node:               node-3.internal/10.0.1.43
Start Time:         Mon, 21 Jul 2026 14:02:11 +0000
Labels:             app=web-backend,version=2.4.1
Status:             Running
IP:                 172.16.0.55
Controlled By:      ReplicaSet/web-backend-6f9d7c4

Containers:
  web-backend:
    Image:          registry.internal/web-backend:2.4.1
    Limits:
      cpu:          500m
      memory:       512Mi
    Requests:
      cpu:          250m
      memory:       256Mi
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    137
      Started:      Mon, 21 Jul 2026 13:58:44 +0000
      Finished:     Mon, 21 Jul 2026 14:01:59 +0000
    Ready:          True
    Restart Count:  7

Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True

Events:
  Type     Reason     Age    From               Message
  ----     ------     ---    ----                -------
  Warning  OOMKilling 4m     kubelet/node-3      Memory cgroup out of memory: Kill process 18423 (node) score 1842 or sacrifice child
  Normal   Pulled     2m     kubelet/node-3      Successfully pulled image registry.internal/web-backend:2.4.1
  Normal   Started    2m     kubelet/node-3      Started container web-backend
"""

SQL_OUTPUT = """\
Query: SELECT session_id, user_id, payload_bytes, created_at
       FROM active_sessions
       ORDER BY payload_bytes DESC
       LIMIT 10;

Results:
session_id                           | user_id | payload_bytes | created_at
-------------------------------------|---------|---------------|-----------------------------
sess_9f2a1c3b-e841-4d2e-b3f1-001    | 84291   | 48432128      | 2026-07-21 13:55:02
sess_7e3b2d4c-f952-5e3f-c4g2-002    | 71038   | 41943040      | 2026-07-21 13:48:17
sess_1a2b3c4d-0123-4567-89ab-003    | 93847   | 39845888      | 2026-07-21 13:51:44
sess_5c6d7e8f-9012-3456-7890-004    | 22819   | 36700160      | 2026-07-21 13:59:11
sess_3d4e5f6a-8901-2345-6789-005    | 57392   | 33554432      | 2026-07-21 13:44:28

Top 5 sessions consume 200,475,648 bytes total (191 MB)
Average payload_bytes across all active sessions: 22,020,096 (21 MB)
Total active sessions: 847

Execution time: 847ms
Rows returned: 10
"""

BASH_OUTPUT = """\
$ df -h /var/lib/kubelet && free -m && top -bn1 | grep node | head -5

Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        80G   74G  6.0G  93% /var/lib/kubelet

              total        used        free      shared  buff/cache   available
Mem:          15872       14901         184         312         786         659
Swap:             0           0           0

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
18423 node      20   0  712432 509244  21344 S  94.3  3.1    2:18.44 node
18891 node      20   0  698112 487932  20112 S  87.1  3.0    1:44.22 node
17234 node      20   0  534912 412800  18934 S  71.2  2.5    0:58.11 node

Exit code: 0
"""

FILE_OUTPUT = """\
# deployment.yaml — web-backend v2.4.1
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-backend
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-backend
  template:
    spec:
      containers:
      - name: web-backend
        image: registry.internal/web-backend:2.4.1
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: SESSION_CACHE_MAX_MB
          value: "unlimited"
        - name: NODE_OPTIONS
          value: "--max-old-space-size=2048"
        - name: DB_POOL_SIZE
          value: "50"
"""

# Simulated tool registry — maps tool call name → output string
TOOL_OUTPUTS = {
    "kubectl_describe_pod": KUBECTL_OUTPUT,
    "sql_query_sessions":   SQL_OUTPUT,
    "bash_check_resources": BASH_OUTPUT,
    "file_read_manifest":   FILE_OUTPUT,
}

# Tool → ContextStream tool_name mapping
TOOL_CS_NAME = {
    "kubectl_describe_pod": "kubectl",
    "sql_query_sessions":   "sql",
    "bash_check_resources": "bash",
    "file_read_manifest":   "file",
}

AGENT_TASK = (
    "Production incident: web-backend pods are OOMKilled repeatedly in the production namespace. "
    "Investigate the root cause and recommend the minimum change to resolve it."
)

DIAGNOSIS_PROMPT_TEMPLATE = """\
You are an SRE agent diagnosing a production incident.

{context}

Based on the evidence above, provide:
1. Root cause (one sentence, specific — name exact resource, metric, value)
2. Contributing factors (bullet list, max 3)
3. Recommended fix (one sentence, specific — what to change and to what value)
"""
