import mapInferredSchema from "@/modules/graphModels/mapInferredSchema";

describe("mapInferredSchema", () => {
  describe("when $defs is missing or empty", () => {
    it("returns an empty entities array", () => {
      const result = mapInferredSchema({});

      expect(result).toEqual({ options: {}, entities: [] });
    });
  });

  describe("entity filtering", () => {
    it("skips entity defs whose key ends in \"Type\" (enum wrappers)", () => {
      const result = mapInferredSchema({
        $defs: {
          PersonType: { title: "PersonType", properties: {} },
          Person: { title: "Person", properties: {} },
        },
      });

      expect(result.entities).toHaveLength(1);
      expect(result.entities[0].name).toBe("Person");
    });

    it("falls back to the $defs key as the entity name when title is absent", () => {
      const result = mapInferredSchema({
        $defs: { Company: { properties: {} } },
      });

      expect(result.entities[0].name).toBe("Company");
    });

    it("defaults description to an empty string when absent", () => {
      const result = mapInferredSchema({
        $defs: { Company: { properties: {} } },
      });

      expect(result.entities[0].description).toBe("");
    });
  });

  describe("field filtering", () => {
    it("drops the is_type and metadata bookkeeping properties", () => {
      const result = mapInferredSchema({
        $defs: {
          Person: {
            properties: {
              is_type: { type: "string" },
              metadata: { type: "string" },
              name: { type: "string" },
            },
          },
        },
      });

      const fieldNames = result.entities[0].fields.map((f) => f.name);
      expect(fieldNames).toEqual(["name"]);
    });
  });

  describe("field kind mapping", () => {
    it("maps a $ref property to a single relation field", () => {
      const result = mapInferredSchema({
        $defs: {
          Person: { properties: { employer: { $ref: "#/$defs/Company" } } },
        },
      });

      expect(result.entities[0].fields[0]).toMatchObject({
        name: "employer",
        kind: "relation",
        relation: { targetEntityName: "Company", cardinality: "one" },
      });
    });

    it("maps an array of $ref items to a many relation field", () => {
      const result = mapInferredSchema({
        $defs: {
          Person: {
            properties: {
              friends: { type: "array", items: { $ref: "#/$defs/Person" } },
            },
          },
        },
      });

      expect(result.entities[0].fields[0]).toMatchObject({
        name: "friends",
        kind: "relation",
        relation: { targetEntityName: "Person", cardinality: "many" },
      });
    });

    it("maps a number/integer type to a primitive number field", () => {
      const result = mapInferredSchema({
        $defs: { Person: { properties: { age: { type: "integer" } } } },
      });

      expect(result.entities[0].fields[0]).toMatchObject({
        name: "age",
        kind: "primitive",
        primitiveType: "number",
      });
    });

    it("maps a boolean type to a primitive boolean field", () => {
      const result = mapInferredSchema({
        $defs: { Person: { properties: { active: { type: "boolean" } } } },
      });

      expect(result.entities[0].fields[0]).toMatchObject({
        kind: "primitive",
        primitiveType: "boolean",
      });
    });

    it("defaults an unrecognized/absent type to a primitive string field", () => {
      const result = mapInferredSchema({
        $defs: { Person: { properties: { name: { type: "string" } } } },
      });

      expect(result.entities[0].fields[0]).toMatchObject({
        kind: "primitive",
        primitiveType: "string",
      });
    });

    it("marks a field required when its name is in the entity's required list", () => {
      const result = mapInferredSchema({
        $defs: {
          Person: {
            properties: { name: { type: "string" } },
            required: ["name"],
          },
        },
      });

      expect(result.entities[0].fields[0].required).toBe(true);
    });

    it("marks a field not required when absent from the entity's required list", () => {
      const result = mapInferredSchema({
        $defs: { Person: { properties: { nickname: { type: "string" } } } },
      });

      expect(result.entities[0].fields[0].required).toBe(false);
    });
  });

  describe("generated ids", () => {
    it("assigns a distinct _id to every entity and every field", () => {
      const result = mapInferredSchema({
        $defs: {
          Person: { properties: { name: { type: "string" }, age: { type: "integer" } } },
          Company: { properties: {} },
        },
      });

      const entityIds = result.entities.map((e) => e._id);
      const fieldIds = result.entities.flatMap((e) => e.fields.map((f) => f._id));
      const allIds = [...entityIds, ...fieldIds];

      expect(new Set(allIds).size).toBe(allIds.length);
    });
  });
});
