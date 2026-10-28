import { structuredRowValues } from "@/modules/business/structuredRow";
import { structuredRowFields } from "@/modules/business/structuredRowFields";
import type { BusinessGraphNode } from "@/modules/business/types";

export default function StructuredRowDetails({ node }: { node: BusinessGraphNode }): React.JSX.Element | null {
  const content = structuredRowValues(node);
  if (!content) return null;
  const fields = structuredRowFields(node, content);
  return (
    <details className="mt-2 text-xs">
      <summary className="cursor-pointer">Row details</summary>
      <div className="mt-2 max-h-60 overflow-auto">
        {fields.length > 0 ? (
          <table className="w-full table-fixed text-left" aria-label="Row values">
            <thead><tr><th scope="col" className="w-2/5 p-1">Column</th><th scope="col" className="p-1">Value</th></tr></thead>
            <tbody>
              {fields.map((field, index) => (
                <tr key={`${field.column}-${index}`}>
                  <th scope="row" className="break-words p-1 align-top font-normal">{field.column}</th>
                  <td className="whitespace-pre-wrap break-words p-1 align-top">
                    {field.value === "" ? <span className="italic">Empty</span> : field.value}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <pre className="whitespace-pre-wrap break-words">{content}</pre>}
        {fields.length > 0 && (
          <details className="mt-2">
            <summary className="cursor-pointer">Original row text</summary>
            <pre className="mt-1 whitespace-pre-wrap break-words">{content}</pre>
          </details>
        )}
      </div>
    </details>
  );
}
