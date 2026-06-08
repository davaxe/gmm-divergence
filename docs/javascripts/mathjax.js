window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    typeset: false
  }
}

document$.subscribe(({ body }) => {
  if (!window.MathJax || !MathJax.typesetPromise) {
    return
  }

  MathJax.startup.promise.then(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear([body])
    MathJax.texReset()
    return MathJax.typesetPromise([body])
  }).catch((err) => {
    console.error("MathJax rendering failed:", err)
  })
})

component$.subscribe(({ ref }) => {
  if (
    window.MathJax &&
    MathJax.typesetPromise &&
    ref.classList.contains("md-annotation")
  ) {
    MathJax.startup.promise
      .then(() => MathJax.typesetPromise([ref]))
      .catch((err) => console.error("MathJax annotation rendering failed:", err))
  }
})
