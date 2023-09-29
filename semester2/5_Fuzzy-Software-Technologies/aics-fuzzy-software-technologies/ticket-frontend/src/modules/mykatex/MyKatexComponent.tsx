import katex from 'katex';
import { useEffect } from 'react';
import { useRef } from 'react';

export interface MyKatexComponentProps {
    latexStr: string;
}

export default function MyKatexComponent({ latexStr }: MyKatexComponentProps) {
    const myRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (myRef.current) {
            katex.render(latexStr, myRef.current, {
                throwOnError: false
            });
        }
    }, [latexStr])

    return (
        <div ref={myRef}>

        </div>
    )
}
