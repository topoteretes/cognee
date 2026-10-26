import { HttpError } from "@/services/http/errors";
import {
  NAME_MAX_LENGTH,
  NAME_REFUSED,
  displayName,
  isProfileNameSource,
  nameProblem,
  normalizeNameInput,
  refusedFieldsFromDetail,
  refusedFieldsFromError,
  sameNames,
} from "../profileName";

describe("normalizeNameInput", () => {
  it("trims and collapses whitespace", () => {
    expect(normalizeNameInput("  Mary   Ann ")).toBe("Mary Ann");
    expect(normalizeNameInput("Mary\t\nAnn")).toBe("Mary Ann");
  });

  it.each([null, undefined, "", "   ", "\n\t"])("reads %p as empty", (value) => {
    expect(normalizeNameInput(value)).toBeNull();
  });

  it("composes a decomposed name, as the backend does", () => {
    // "e" followed by a combining diaeresis is the same name as the single
    // character, and must be the same string and the same length.
    expect(normalizeNameInput("Zoë")).toBe("Zoë");
  });
});

describe("nameProblem", () => {
  it("accepts the limit and refuses one over, counted in code points", () => {
    expect(nameProblem("a".repeat(NAME_MAX_LENGTH))).toBeNull();
    expect(nameProblem("a".repeat(NAME_MAX_LENGTH + 1))).toBe("too_long");
    // 100 astral characters are 200 UTF-16 units and exactly the limit.
    expect(nameProblem("\u{1F642}".repeat(NAME_MAX_LENGTH))).toBeNull();
  });

  it("measures after composition, so a long decomposed name is not refused for its parts", () => {
    // 51 letters, each written as two code points: 102 decomposed, 51 composed.
    const decomposed = "é".repeat(51);

    expect(Array.from(decomposed).length).toBe(102);
    expect(nameProblem(decomposed)).toBeNull();
  });

  it("measures after whitespace is collapsed, not before", () => {
    expect(nameProblem(` ${"a".repeat(NAME_MAX_LENGTH)} `)).toBeNull();
  });

  it("does not treat an empty name as a problem: it clears", () => {
    expect(nameProblem("")).toBeNull();
    expect(nameProblem(null)).toBeNull();
  });
});

describe("sameNames", () => {
  it("ignores whitespace and composition", () => {
    expect(
      sameNames(
        { firstName: " Zoë ", lastName: "Lovelace" },
        { firstName: "Zoë", lastName: " Lovelace" },
      ),
    ).toBe(true);
  });

  it("treats empty and null alike, and notices a real change", () => {
    expect(sameNames({ firstName: "", lastName: null }, { firstName: null, lastName: "  " })).toBe(true);
    expect(sameNames({ firstName: "Ada", lastName: null }, { firstName: "Ada", lastName: "L" })).toBe(false);
  });
});

describe("displayName", () => {
  it("joins the names that are set", () => {
    expect(displayName({ firstName: "Ada", lastName: "Lovelace" }, "x")).toBe("Ada Lovelace");
    expect(displayName({ firstName: "Ada", lastName: null }, "x")).toBe("Ada");
    expect(displayName({ firstName: null, lastName: "Lovelace" }, "x")).toBe("Lovelace");
  });

  it("falls back, and to an empty string rather than the string 'undefined'", () => {
    expect(displayName({ firstName: null, lastName: null }, "From Auth0")).toBe("From Auth0");
    expect(displayName(null, "From Auth0")).toBe("From Auth0");
    expect(displayName(null, undefined)).toBe("");
  });
});

describe("isProfileNameSource", () => {
  it("accepts settings and nothing else, onboarding included", () => {
    expect(isProfileNameSource("settings")).toBe(true);
    expect(isProfileNameSource("onboarding")).toBe(false);
    expect(isProfileNameSource(undefined)).toBe(false);
  });
});

describe("refusedFieldsFromDetail", () => {
  const entry = (field: string) => ({
    type: "value_error",
    loc: ["body", field],
    msg: "Value error, Name contains characters that are not allowed",
    input: "the typed name",
  });

  it.each([
    ["firstName", ["firstName"]],
    ["first_name", ["firstName"]],
    ["lastName", ["lastName"]],
    ["last_name", ["lastName"]],
  ])("reads %s", (field, expected) => {
    expect(refusedFieldsFromDetail([entry(field)])).toEqual(expected);
  });

  it("reads both when both were refused, once each", () => {
    expect(refusedFieldsFromDetail([entry("firstName"), entry("lastName"), entry("firstName")])).toEqual([
      "firstName",
      "lastName",
    ]);
  });

  it("ignores a location it does not know, and anything that is not a detail list", () => {
    expect(refusedFieldsFromDetail([entry("source")])).toEqual([]);
    expect(refusedFieldsFromDetail([{ loc: [] }, { loc: "firstName" }, null, 7])).toEqual([]);
    expect(refusedFieldsFromDetail("Name is bad")).toEqual([]);
    expect(refusedFieldsFromDetail(undefined)).toEqual([]);
  });

  it("returns names of fields only, never the text the person typed", () => {
    expect(JSON.stringify(refusedFieldsFromDetail([entry("firstName")]))).not.toContain("typed name");
  });
});

describe("refusedFieldsFromError", () => {
  const refusal = (body: unknown, status = 422) =>
    new HttpError(status, "Unprocessable Entity", "name_refused", body as never);

  it("reads the fields the route named", () => {
    expect(refusedFieldsFromError(refusal({ error: NAME_REFUSED, fields: ["lastName"] }))).toEqual(["lastName"]);
  });

  it("says both when the route could not say which", () => {
    // A message under both is true of whichever name it was, which a message
    // under only one would not be.
    expect(refusedFieldsFromError(refusal({ error: NAME_REFUSED, fields: [] }))).toEqual(["firstName", "lastName"]);
    expect(refusedFieldsFromError(refusal({ error: NAME_REFUSED }))).toEqual(["firstName", "lastName"]);
  });

  it("ignores a field it does not know", () => {
    expect(refusedFieldsFromError(refusal({ error: NAME_REFUSED, fields: ["email", "lastName"] }))).toEqual(["lastName"]);
  });

  it.each([
    ["a 422 that is not this route's", new HttpError(422, "x", "x", { error: "something_else" })],
    ["a 422 with a text body", new HttpError(422, "x", "x", "plain text")],
    ["another status", refusal({ error: NAME_REFUSED }, 502)],
    ["something that is not an HttpError", new Error("down")],
    ["nothing", undefined],
  ])("is not a refusal for %s", (_label, error) => {
    expect(refusedFieldsFromError(error)).toBeNull();
  });
});
