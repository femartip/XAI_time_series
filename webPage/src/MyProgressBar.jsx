import ProgressBar from "@ramonak/react-progress-bar";

const BasicExample = ({ currValue }) => {
  return <div>
    <ProgressBar completed={currValue} customLabel={"AI classification confidence: " + currValue.toString() + "%"} />
  </div>
}

export default BasicExample