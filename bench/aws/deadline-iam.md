# Independent stop deadline

The safe runner requires an EventBridge Scheduler execution role before renting. Guest
shutdown and laptop cleanup do not cover simultaneous guest failure and laptop loss.
The one-time AWS deadline requests a forced EC2 stop, then deletes itself. It does not
terminate the instance or delete retained EBS data. Forced stop can require filesystem repair.

## Account setup

An account administrator must provision the role and caller permissions. Keep normal runs
under `arn:aws:iam::123456789012:user/bench-runner`; do not switch the runner to root.
No IAM changes are performed by these scripts.

Recommended role: `arn:aws:iam::123456789012:role/g16-benchmark-stop`.
Default region and schedule group: `us-east-1`, `default`.

Role trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "scheduler.amazonaws.com"},
    "Action": "sts:AssumeRole",
    "Condition": {"StringEquals": {
      "aws:SourceAccount": "123456789012",
      "aws:SourceArn": "arn:aws:scheduler:us-east-1:123456789012:schedule-group/default"
    }}
  }]
}
```

Role permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "ec2:StopInstances",
    "Resource": "arn:aws:ec2:us-east-1:123456789012:instance/*",
    "Condition": {"StringEquals": {"ec2:ResourceTag/Owner": "bench-runner"}}
  }]
}
```

Caller managed policy `G16BenchmarkScheduler` is attached to `bench-runner`.
Scheduler access is unrestricted as requested; PassRole remains limited to this role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SchedulerFullAccess",
      "Effect": "Allow",
      "Action": "scheduler:*",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::123456789012:role/g16-benchmark-stop",
      "Condition": {"StringEquals": {"iam:PassedToService": "scheduler.amazonaws.com"}}
    }
  ]
}
```

Set `G16_AWS_STOP_ROLE_ARN` to the provisioned role. For another region or group, update
the role permission scope and role trust first. `aws:SourceArn` in role trust must name
the schedule group, not an individual schedule. Scheduler read access alone does not
prove CreateSchedule, PassRole or target execution will succeed.

## Validation gate

Before accepting large-run safety, validate a real short deadline against an owned disposable
instance and confirm EC2 reaches stopped. Separately validate the guest idle and hard timers.
Synthetic request/entrypoint tests do not establish service execution, role trust, systemd
installation or EC2 shutdown behavior. No large fixture upload until those checks pass.

The runner records the schedule alongside instance results. Keep the deadline armed through
local failure and SIGKILL; completion cleanup still stops immediately rather than waiting.
A hard deadline is an AWS control-plane backstop, not a guarantee against AWS service outages.

Sources:
- [Universal targets](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-universal.html)
- [Execution role setup](https://docs.aws.amazon.com/scheduler/latest/UserGuide/setting-up.html)
- [Trust-policy confused deputy protection](https://docs.aws.amazon.com/scheduler/latest/UserGuide/cross-service-confused-deputy-prevention.html)
