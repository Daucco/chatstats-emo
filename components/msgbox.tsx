const MsgBox = ({message, setMessage}: any) => {
    /*const [postMsg, setMsg] = useState("Write something awesome! :D");*/
    
    return (
        <textarea
            rows={3}
            cols={30}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
        />
    );
};

export default MsgBox;