import type { CogneeInstance } from "../../instances/types";
import getSkills, { SKILLS_PAGE_LIMIT, getSkillsCount } from "../getSkills";

const fetch = jest.fn();
const instance = { name: "test", instanceId: "test", fetch } as CogneeInstance;
const response = (body: unknown, ok = true) =>
  ({ ok, status: ok ? 200 : 404, json: async () => body }) as Response;

const rows = (length: number) =>
  Array.from({ length }, (_, i) => ({ id: String(i), name: `skill-${i}` }));

const urlFor = (fragment: string) =>
  fetch.mock.calls.map((call) => String(call[0])).find((url) => url.includes(fragment));

beforeEach(() => fetch.mockReset());

test("asks for the largest page the server will serve", async () => {
  fetch.mockResolvedValue(response([]));
  await getSkills(instance, "ds-1");
  expect(urlFor("/v1/skills/?")).toContain(`limit=${SKILLS_PAGE_LIMIT}`);
});

test("reports the dataset total, not the page size", async () => {
  // The regression from #4279: 378 skills, a page that stops at the cap.
  fetch.mockImplementation((url: string) =>
    Promise.resolve(url.includes("/count") ? response({ count: 378 }) : response(rows(200))),
  );

  const page = await getSkills(instance, "ds-1");

  expect(page.skills).toHaveLength(200);
  expect(page.total).toBe(378);
});

test("a full listing is not reported as truncated", async () => {
  fetch.mockImplementation((url: string) =>
    Promise.resolve(url.includes("/count") ? response({ count: 12 }) : response(rows(12))),
  );

  const page = await getSkills(instance, "ds-1");

  expect(page.total).toBe(page.skills.length);
});

test("a pod without /count never reports fewer skills than it returned", async () => {
  // An older server 404s the count route; the rows it did return are still the
  // floor for the total, so the page cannot claim "12 of 0".
  fetch.mockImplementation((url: string) =>
    url.includes("/count") ? Promise.reject(new Error("404")) : Promise.resolve(response(rows(12))),
  );

  const page = await getSkills(instance, "ds-1");

  expect(page.skills).toHaveLength(12);
  expect(page.total).toBe(12);
});

test("counts scope to the dataset and honour include_inactive", async () => {
  fetch.mockResolvedValue(response({ count: 5 }));

  await expect(getSkillsCount(instance, "ds-1", true)).resolves.toBe(5);
  const url = urlFor("/v1/skills/count");
  expect(url).toContain("dataset_id=ds-1");
  expect(url).toContain("include_inactive=true");
});

test.each([{ count: -1 }, { count: 1.5 }, {}, { count: "100" }])(
  "a malformed count is not trusted: %j",
  async (body) => {
    fetch.mockResolvedValue(response(body));
    await expect(getSkillsCount(instance, "ds-1")).resolves.toBe(0);
  },
);

test("a failing listing still rejects", async () => {
  fetch.mockImplementation((url: string) =>
    url.includes("/count")
      ? Promise.resolve(response({ count: 0 }))
      : Promise.reject(new Error("boom")),
  );
  await expect(getSkills(instance, "ds-1")).rejects.toThrow();
});
