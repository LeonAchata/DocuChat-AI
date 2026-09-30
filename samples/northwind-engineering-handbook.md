# Northwind Labs Engineering Handbook

This handbook describes how the Northwind Labs engineering team ships software. It is a fictional
document written as sample data for DocuChat.

## Development workflow

### Branching and reviews

All work happens on short-lived feature branches created from `main`. A pull request needs at
least one approving review from a code owner before it can be merged. Pull requests that touch
billing, authentication or data retention need two approvals, one of them from the security
guild.

Pull requests should stay under 400 changed lines. Larger changes must be split or hidden behind
a feature flag so they can be reviewed incrementally.

### Continuous integration

Every push runs linting, type checking, unit tests and a container build. The main branch must
stay green: if a merge breaks the build, the author either fixes it within 30 minutes or reverts
the change.

## Deployments

### Release process

Production deploys happen continuously from `main` through a progressive rollout: 5% of traffic
for 15 minutes, then 25%, then 100%. The rollout pauses automatically if the error rate rises
above 1% or p95 latency grows by more than 20% compared with the previous release.

Deploys are frozen from December 20 to January 5 and during major customer events announced in
the #release-calendar channel. Emergency fixes during a freeze need approval from the on-call
engineering manager.

### Rollbacks

Any engineer may roll back a release without asking for permission. Rollbacks take under two
minutes because every release keeps the previous container image warm.

## On-call

### Rotation

Each product team runs a weekly on-call rotation that starts on Monday at 10:00 UTC. Engineers
join the rotation after three months at the company and after shadowing one full rotation.

On-call engineers receive a stipend of 300 USD per week, plus 50 USD for every page received
between 22:00 and 07:00 local time.

### Incident response

Pages must be acknowledged within 5 minutes. Incidents are classified by severity:

- SEV1: the product is down or data is at risk for many customers. Response within 15 minutes,
  status page updated every 30 minutes.
- SEV2: a major feature is degraded. Response within 1 hour.
- SEV3: minor impact with a workaround available. Handled during business hours.

Every SEV1 and SEV2 incident gets a blameless postmortem, published within five business days.

## Security

Production access is granted just-in-time for a maximum of four hours and requires hardware-key
multi-factor authentication. Secrets live in the central vault and are rotated every 90 days;
committing a secret to a repository is treated as a SEV2 incident.

Customer data must never be copied to laptops or personal cloud storage. Analytics on customer
data run inside the data warehouse, where personally identifiable fields are masked by default.

## Time off

Engineers get 28 days of paid time off per year in addition to public holidays. Up to 5 unused
days can be carried over into the first quarter of the following year. Time off longer than two
consecutive weeks should be requested at least one month in advance.
