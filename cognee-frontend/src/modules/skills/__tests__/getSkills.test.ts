import type { CogneeInstance } from "../../instances/types";
import getSkills from "../getSkills";
import type { SkillRaw } from "../types";

const fetch = jest.fn();
const instance = { name: "test", instanceId: "test", fetch } as CogneeInstance;
const response = (body: unknown) => ({ json: async () => body }) as Response;
const skills = (start: number, end: number): SkillRaw[] =>
  Array.from({ length: end - start }, (_, index) => ({
    id: String(start + index),
    name: `Skill ${start + index}`,
  }));

beforeEach(() => fetch.mockReset());

test("requests the backend maximum instead of silently stopping at 200 skills", async () => {
  fetch.mockResolvedValue(response(skills(0, 378)));

  await expect(getSkills(instance, "dataset")).resolves.toHaveLength(378);
  expect(fetch.mock.calls[0][0]).toContain("limit=1000&offset=0");
});

test("traverses every skills page", async () => {
  fetch
    .mockResolvedValueOnce(response(skills(0, 1000)))
    .mockResolvedValueOnce(response(skills(1000, 1001)));

  await expect(getSkills(instance, "dataset", true)).resolves.toHaveLength(1001);
  expect(fetch.mock.calls[1][0]).toContain("limit=1000&offset=1000");
  expect(fetch.mock.calls[1][0]).toContain("include_inactive=true");
});

test("stops when an old server ignores pagination", async () => {
  fetch.mockResolvedValue(response(skills(0, 1000)));

  await expect(getSkills(instance, "dataset")).rejects.toThrow("pagination");
  expect(fetch).toHaveBeenCalledTimes(2);
});
