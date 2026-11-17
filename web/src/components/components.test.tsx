import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { toPayload } from "@/app/console/playground/playground";
import { ContributionBars } from "@/components/contributions";
import { DecisionBadge, ProbabilityBar } from "@/components/ui";

describe("ContributionBars", () => {
  it("orders by size and says which way each feature pushes", () => {
    render(
      <ContributionBars
        contributions={[
          ["amt_log", -0.4],
          ["graph_2hop_fraud_cards", 1.2],
          ["C1", 0.1],
        ]}
        values={{ amt_log: 5.5, graph_2hop_fraud_cards: 3, C1: 1 }}
      />,
    );
    const rows = screen.getAllByRole("listitem");
    expect(rows.map((r) => r.getAttribute("aria-label")?.split(" =")[0])).toEqual(["graph_2hop_fraud_cards", "amt_log", "C1"]);
    expect(rows[0]?.getAttribute("aria-label")).toContain("raises risk");
    expect(rows[1]?.getAttribute("aria-label")).toContain("lowers risk");
  });
});

describe("badges", () => {
  it("never relies on colour alone", () => {
    render(<DecisionBadge decision="decline" />);
    expect(screen.getByText("Decline")).toBeTruthy();
  });
  it("exposes the probability as a meter", () => {
    render(<ProbabilityBar p={0.37} />);
    expect(screen.getByRole("meter").getAttribute("aria-valuenow")).toBe("37");
  });
});

describe("playground payload", () => {
  it("sends empty fields as null and numbers as numbers", () => {
    const payload = toPayload({
      TransactionAmt: "19.99",
      ProductCD: "W",
      card1: "",
      card4: "visa",
      card6: "",
      addr1: "315",
      addr2: "",
      P_emaildomain: " gmail.com ",
      DeviceType: "",
      DeviceInfo: "",
      D1: "0",
    });
    expect(payload).toMatchObject({ TransactionAmt: 19.99, card1: null, addr1: 315, card6: null, P_emaildomain: "gmail.com", D1: 0 });
  });
});
