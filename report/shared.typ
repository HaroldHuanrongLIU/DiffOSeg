#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/tokens.typ": um-blue, um-blue-light, um-grey
#let tables = json("../results/course_project/tables.json")
#let course-table(headers, rows, size: 10pt, inset: 5pt) = block(width: 100%)[
  #set text(size: size)
  #table(
    columns: headers.len(),
    inset: inset,
    stroke: .4pt + um-grey,
    fill: (x, y) => if y == 0 { um-blue } else { white },
    table.header(..headers.map(h => text(fill: white, weight: "bold", h))),
    ..rows.flatten().map(cell => [#cell]),
  )
]
