import React from "react";

function breadcrumbParts(relativePath) {
  if (!relativePath || relativePath === ".") {
    return ["."];
  }
  return relativePath.split("/").filter(Boolean);
}

export function BrowserView({ entry, relativePath }) {
  const isFile = entry.type === "file";
  const title = isFile ? entry.data.name : relativePath;

  return (
    <main className="shell">
      <section className="panel">
        <div className="panel-head">
          <h2>{isFile ? entry.data.name : `Browse ${relativePath}`}</h2>
          <a href="/">Back to dashboard</a>
        </div>

        <p className="subtitle">{relativePath}</p>

        {!isFile ? (
          <div className="browser">
            <div className="browser-toolbar">
              <span>{relativePath}</span>
              <a href="/browse?path=.">Root</a>
            </div>
            <div className="breadcrumbs">
              {breadcrumbParts(relativePath).map((part, index, parts) => {
                const pathValue = parts.slice(0, index + 1).join("/");
                return (
                  <a key={`${part}-${index}`} href={`/browse?path=${encodeURIComponent(pathValue || ".")}`}>
                    {part}
                  </a>
                );
              })}
            </div>
            <table className="listing">
              <thead>
                <tr>
                  <th>Kind</th>
                  <th>Name</th>
                  <th>Size</th>
                  <th>Modified</th>
                </tr>
              </thead>
              <tbody>
                {entry.data.map((child) => (
                  <tr key={child.relativePath}>
                    <td>{child.kind}</td>
                    <td>
                      <a href={child.href}>{child.name}</a>
                    </td>
                    <td>{child.size}</td>
                    <td>{child.modified}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="browser">
            <div className="browser-toolbar">
              <span>{title}</span>
              <a href={`/browse?path=${encodeURIComponent(entry.data.parentPath || ".")}`}>Up one level</a>
            </div>
            <p className="subtitle">
              {entry.data.relativePath} · {entry.data.size} · {entry.data.modified}
            </p>
            <pre className="code">{entry.data.preview.join("\n")}</pre>
          </div>
        )}
      </section>
    </main>
  );
}
