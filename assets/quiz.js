document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-choice]");
  if (!button) return;

  const quiz = button.closest(".quiz");
  const result = quiz.querySelector(".result");
  const ok = button.dataset.choice === quiz.dataset.answer;
  result.textContent = ok ? "Correct. Keep that shape in memory." : "Not quite. Re-read the two choices and try again.";
});
