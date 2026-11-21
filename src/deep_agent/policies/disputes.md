# Chargebacks and disputes

- A CHARGEBACK with status OPEN means the customer's bank has opened a dispute and has already
  withdrawn the disputed amount from CloudOrder.
- While a dispute is open, never issue a refund on that order. A refund would pay the customer
  twice.
- Escalate to the disputes queue with the dispute ID and a short summary. Use priority high
  when the respondBy date is within 14 days.
- Tell the customer that their bank's dispute is in progress, that the bank decides the outcome
  and returns the funds if it rules in their favor, and that CloudOrder's disputes team is
  handling the case. Do not discuss how CloudOrder will respond to the dispute.
