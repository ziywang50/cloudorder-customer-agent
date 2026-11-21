# Refund policy

Every refund needs a reason code. The refund rules engine (`check_refund_eligibility`) is
authoritative for eligibility and limits; this document explains the rules it applies.

## duplicate_charge
- A duplicate charge exists only when successful captures on an order exceed the order total.
- Refund the extra captured amount in full. Do not refund the order itself.
- Authorization holds, including voided ones, are not charges. See payments.md.

## lost_package
- A shipment is lost when it is undelivered, past its promised delivery date, and has had no
  carrier scan for at least 5 days.
- Refund the price plus tax of every item in the lost shipment. Shipping is refunded only if
  the whole order was in lost shipments.
- Items in shipments that were delivered are not covered.

## damaged_item_keep
- For cosmetic damage where the item still works and the customer prefers to keep it, offer a
  partial refund of up to 25% of the item's price plus tax.
- Functional defects are not handled with this reason: escalate to tier2_support for a
  replacement or return.

## Always
- No refund while a chargeback is open on the order. See disputes.md.
- Never exceed maxForReasonCents or what is still refundable on the order.
- A human reviewer approves every refund before it is issued and may lower the amount.
- Refunds go back to the original payment method and typically settle within 7 days.
